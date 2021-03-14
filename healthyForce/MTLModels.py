import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MTL:
    def __init__(self):
        pass

    def aggregate_losses(self, losses):
        pass

    def adjust_after_validation(self, losses, epoch):
        pass


class MTLRandom(MTL):
    def __init__(self, ntasks, verbose=1):
        self.ntasks = ntasks

    def aggregate_losses(self, losses):
        return losses[np.random.randint(self.ntasks)]

    def adjust_after_validation(self, losses, epoch):
        return


class MTLUncertanty(MTL):

    def __init__(self, ntasks):

        super(MTLUncertanty, self).__init__()

        self.ntasks = ntasks
        # We have to be set in the Lightning Module
        #self.logsigma = nn.Parameter(torch.zeros(self.ntasks))
        self.logsigma = None

    def aggregate_losses(self, losses):
        """
            Input: a list/set/dict of losses
            Output: a single value
        """
        total_loss = 0
        for i, l in enumerate(losses):
            total_loss = total_loss + (l / (2. * torch.exp(self.logsigma[i])) + (self.logsigma[i]/2.))
        return total_loss

    def adjust_after_validation(self, losses, epoch):
        return

class MTLEqual(MTL):

    def __init__(self, ntasks):

        super(MTLEqual, self).__init__()
        self.ntasks = ntasks

    def aggregate_losses(self, losses):
        return sum(losses) / self.ntasks

    def adjust_after_validation(self, losses, epoch):
        return

class MTLDWA(MTL):

    def __init__(self, ntasks, algorithm, temperature=2, min_epochs_to_start=2, verbose=1):

        super(MTLDWA, self).__init__()
        self.ntasks = ntasks
        self.lambda_weight = torch.ones(self.ntasks)
        self.loss_t_1 = torch.ones(self.ntasks)
        self.loss_t_2 = torch.ones(self.ntasks)
        self.temperature = torch.ones(1) * temperature
        self.min_epochs_to_start = min_epochs_to_start

        self.algorithm = algorithm

        # Variables for ewa and trend version of DWA
        self.verbose = verbose
        self.max_epochs = 100
        self.history = torch.zeros(self.ntasks, self.max_epochs)
        self.winsize = 3

        #data = np.array([0,1,200,300,-10,20,10,-20,10,-20,1000])
        #ewma(data, 5, 0.9), trend(data[5:10])


    def aggregate_losses(self, losses):
        total_loss = 0
        #self.lambda_weight = self.lambda_weight.type_as(losses[0])
        for i, l in enumerate(losses):
            total_loss += (self.lambda_weight[i] * l)
        return total_loss / self.ntasks

    def adjust_after_validation(self, losses, epoch):

        for i in range(self.ntasks):
            self.loss_t_2[i] = self.loss_t_1[i]
            self.loss_t_1[i] = losses[i].item()

        if epoch >= self.min_epochs_to_start:

            if self.algorithm != "default":
                saved_from_epoch = epoch - self.min_epochs_to_start

            w = {}
            denominator = 0

            for i in range(self.ntasks):
                if self.algorithm == "default":
                    w[i] = min(80., self.loss_t_1[i] / self.loss_t_2[i])
                else:
                    self.history[i][saved_from_epoch] = min(80., self.loss_t_1[i] / self.loss_t_2[i])
                    if self.algorithm == "trend":
                        w[i] = trend(self.history[i][max(0, saved_from_epoch-self.winsize):saved_from_epoch])
                        print("values:", self.history[i][max(0, saved_from_epoch-self.winsize):saved_from_epoch])
                    elif self.algorithm == "ewma":
                        # Todo: need to implement a torch version of it
                        w[i] = trend(self.history[i][max(0, saved_from_epoch-self.winsize):saved_from_epoch])

                if self.verbose > 0:
                    print("w(%d) = %.4f" % (i, w[i]))

                denominator += torch.exp(w[i]/self.temperature)

            for i in range(self.ntasks):
                numerator = self.ntasks * torch.exp(w[i]/self.temperature)
                self.lambda_weight[i] = numerator / denominator

        if self.verbose > 0:
            for i in range(self.ntasks):
                print("Lambda (%d) = %.4f" % (i, self.lambda_weight[i]))


class MTLBandit(MTL):

    def __init__(self, ntasks,
                 # "bandit_alg_weight_assignment"
                 # algorithm: [ucb, ducb]
                 # reward method: [l1/l2, l2/l1, l2-l1]
                 # loss_assignment: ["one", "priority", "all"]
                 strategy="bandit_ucb_l1l2_one",
                 min_epochs_to_start=2, verbose=1):

        super(MTLBandit, self).__init__()

        self.ntasks = ntasks

        self.bandit_alg = strategy.split("_")[1]
        self.bandit_reward_method = strategy.split("_")[2]
        self.bandit_loss_assignment = strategy.split("_")[3]

        self.lambda_weight = torch.ones(self.ntasks)
        self.loss_t_1 = torch.ones(self.ntasks)
        self.loss_t_2 = torch.ones(self.ntasks)

        self.max_epochs = 100
        self.current_weight = torch.zeros(self.ntasks)
        self.reward = torch.zeros(self.max_epochs, self.ntasks)
        self.counts = torch.zeros(self.ntasks)
        self.chosen = torch.zeros(self.max_epochs, self.ntasks)

        self.gammas = torch.zeros(self.max_epochs) + 0.99

        self.min_epochs_to_start = min_epochs_to_start

        self.verbose = verbose

    def aggregate_losses(self, losses):
        total_loss = 0
        for i, l in enumerate(losses):
            total_loss += ((self.lambda_weight[i] * l) / self.ntasks)
        return total_loss

    def adjust_after_validation(self, losses, epoch):

        print("Current epoch:", epoch)

        selected_task_i = -1

        for i in range(self.ntasks):
            self.loss_t_2[i] = self.loss_t_1[i]
            self.loss_t_1[i] = losses[i].item()

            if self.bandit_reward_method == "l1l2":
                self.reward[epoch][i] = min(80., self.loss_t_1[i] / self.loss_t_2[i])
            elif self.bandit_reward_method == "l2l1":
                self.reward[epoch][i] = min(80., self.loss_t_2[i] / self.loss_t_1[i])
            elif self.bandit_reward_method == "l2-l1":
                self.reward[epoch][i] = min(80., self.loss_t_2[i] - self.loss_t_1[i])

        if epoch >= self.min_epochs_to_start:

            if self.bandit_alg == "ducb":
                t_minus_s = get_t_minus_s(self.max_epochs, epoch)
                discount = self.gammas ** t_minus_s
                n_t_gamma = 0
                for i in range(self.ntasks):
                    n_t_gamma += (discount * self.chosen[:, i]).sum()


            # TODO: I could replace this 'for' by a vectorized operation.
            for i in range(self.ntasks):
                # UBC1
                if self.bandit_alg == "ucb":

                    avg_reward = (self.chosen[:, i] * self.reward[:, i]).sum() / self.chosen[:, i].sum()
                    padding = np.sqrt(2.0 * np.log(epoch+1) / (1 + self.counts[i]))
                    self.current_weight[i] =  avg_reward + padding

                # discounted UBC -- very inefficient. Needs improvement
                elif self.bandit_alg == "ducb":

                    N_t_gamma = (discount * self.chosen[:, i]).sum()
                    avg_reward = (discount * self.reward[:, i]).sum() / N_t_gamma

                    padding = 2.0 * np.sqrt(np.log(n_t_gamma)/N_t_gamma)

                    self.current_weight[i] =  avg_reward + padding

                else:
                    print("Unkonwn bandit algorithm %s. Options are 'ubc' and 'ducb'" % (self.bandit_alg))


                if self.verbose > 0:
                    print("Current Reward(%d): %.3f (%.3f + %.3f)" % (i,
                                                          self.current_weight[i],
                                                          avg_reward,
                                                          padding
                                                          )
                          )
            selected_task_i = torch.argmax(self.current_weight).item()
            self.counts[selected_task_i] += 1
            self.chosen[epoch][selected_task_i] = 1

            if self.bandit_loss_assignment == "all":
                for x in range(self.ntasks):
                    self.lambda_weight[x] = self.current_weight[x]

            elif self.bandit_loss_assignment in ["one", "priority"]:
                self.lambda_weight[selected_task_i] = 1
                for task_j in range(self.ntasks):
                    if task_j != selected_task_i:
                        if self.bandit_loss_assignment == "priority":
                            self.lambda_weight[task_j] = 0.5
                        else:
                            self.lambda_weight[task_j] = 0.0

        else:
            # In case the algorithm has not started yet, we are "choosing" all arms
            for x in range(self.ntasks):
                self.chosen[epoch][x] = 1


        if self.verbose > 0:
            print("Selected Task:", selected_task_i)
            for i in range(self.ntasks):
                print("W(%d): %.3f, Counts(%d): %d" % (i, self.current_weight[i], i, self.counts[i]))
            for i in range(self.ntasks):
                print("Lambdas (%d) = %.4f" % (i, self.lambda_weight[i]))
