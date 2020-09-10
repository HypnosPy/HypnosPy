import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd
from datetime import datetime, timedelta
import calendar
import seaborn as sns


from hypnospy import Wearable
from hypnospy import Experiment


class Viewer(object):

    def __init__(self, input: {Wearable, Experiment}):

        if type(input) is Wearable:
            self.wearables = [input]
        elif type(input) is Experiment:
            self.wearables = input.get_all_wearables()


    def view_signals(self, signal_categories: list = ["activity", "hr", "pa_intensity", "sleep"],
                     other_signals: list = [], signal_as_area: list = [], frequency: str = "60S", sleep_cols: str = None,
                     select_days: list = None, colors: list = [], alpha: float = 1.0,
                     zoom: tuple = ("00:00:00", "23:59:59")
                     ):

        for wearable in self.wearables:
            Viewer.view_signals_wearable(wearable, signal_categories, other_signals, signal_as_area, frequency,
                                         sleep_cols, select_days, colors, alpha, zoom)


    @staticmethod
    def view_signals_wearable(wearable: Wearable, signal_categories: list, other_signals: list, signal_as_area: list,
                              frequency: str, sleep_cols: str, select_days: list, colors: list, alpha: float,
                              zoom: tuple):

        sns.set_context("talk", font_scale=1.3, rc={"axes.linewidth": 2, 'image.cmap': 'plasma', })
        plt.rc('font', family='serif')

        cols = []

        for signal in signal_categories:
            if signal == "activity":
                cols.append(wearable.get_activity_col())

            elif signal == "hr":
                if wearable.get_hr_col():
                    cols.append(wearable.get_hr_col())
                else:
                    raise KeyError("HR is not available for PID %s" % wearable.get_pid())

            elif signal == "pa_intensity":
                if hasattr(wearable, 'pa_intensity_cols'):
                    for pa in wearable.pa_intensity_cols:
                        if pa in wearable.data.keys():
                            cols.append(pa)

            elif signal == "sleep":
                for sleep_col in sleep_cols:
                    if sleep_col not in wearable.data.keys():
                        raise ValueError("Could not find sleep_col (%s). Aborting." % sleep_col)
                    cols.append(sleep_col)

            elif signal == "diary" and wearable.diary_onset in wearable.data.keys() and \
                    wearable.diary_offset in wearable.data.keys():
                cols.append(wearable.diary_onset)
                cols.append(wearable.diary_offset)

            else:
                cols.append(signal)

        if len(cols) == 0:
            raise ValueError("Aborting: Empty list of signals to show.")

        cols.append(wearable.time_col)
        cols.append(wearable.experiment_day_col)
        for col in set(other_signals + signal_as_area):
            cols.append(col)

        df_plot = wearable.data[cols].set_index(wearable.time_col).resample(frequency).sum()

        if zoom is not None:
            df_plot = df_plot.between_time(zoom[0], zoom[1], include_start=True, include_end=True)

        # Daily version
        # dfs_per_day = [pd.DataFrame(group[1]) for group in df_plot.groupby(df_plot.index.day)]
        # Based on the experiment day gives us the correct chronological order of the days
        if select_days is not None:
            df_plot = df_plot[df_plot[wearable.experiment_day_col].isin(select_days)]
            if df_plot.empty:
                raise ValueError("Invalid day selection: no remaining data to show.")

        dfs_per_day = [pd.DataFrame(group[1]) for group in df_plot.groupby(wearable.experiment_day_col)]

        fig, ax1 = plt.subplots(len(dfs_per_day), 1, figsize=(14, 8))

        if len(dfs_per_day) == 1:
            ax1 = [ax1]

        plt.rcParams['font.size'] = 18
        plt.rcParams['image.cmap'] = 'plasma'
        plt.rcParams['axes.linewidth'] = 2
        plt.rc('font', family='serif')

        for idx in range(len(dfs_per_day)):
            maxy = 2

            df2_h = dfs_per_day[idx]
            ax1[idx].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, rotation=0)
            ax1[idx].set_facecolor('snow')

            if zoom[0] == "00:00:00" and zoom[1] == "23:59:59": # Default options, we use hour_start_experiment
                new_start_datetime = dfs_per_day[idx].index[0] - timedelta(hours=dfs_per_day[idx].index[0].hour - wearable.hour_start_experiment,
                                                      minutes=dfs_per_day[idx].index[0].minute, seconds=dfs_per_day[idx].index[0].second),
                new_end_datetime = dfs_per_day[idx].index[0] - timedelta(hours=dfs_per_day[idx].index[0].hour - wearable.hour_start_experiment,
                                                      minutes=dfs_per_day[idx].index[0].minute, seconds=dfs_per_day[idx].index[0].second) + timedelta(minutes=1439)

            else:
                new_start_date = dfs_per_day[idx].index[0].date()
                new_start_datetime = datetime(new_start_date.year, new_start_date.month, new_start_date.day, int(zoom[0][0:2]), int(zoom[0][3:5]), int(zoom[0][6:8]))
                new_start_datetime = pd.to_datetime(new_start_datetime)

                new_end_date = dfs_per_day[idx].index[-1].date()
                new_end_datetime = datetime(new_end_date.year, new_end_date.month, new_end_date.day, int(zoom[1][0:2]), int(zoom[1][3:5]), int(zoom[1][6:8]))
                new_end_datetime = pd.to_datetime(new_end_datetime)

            ax1[idx].set_xlim(new_start_datetime, new_end_datetime)

            if "activity" in signal_categories:
                maxy = max(maxy, df2_h[wearable.get_activity_col()].max())
                ax1[idx].plot(df2_h.index, df2_h[wearable.get_activity_col()], label='Activity', linewidth=2,
                              color='black', alpha=alpha)

            if "pa_intensity" in signal_categories:
                ax1[idx].fill_between(df2_h.index, 0, maxy, where=df2_h['hyp_vpa'], facecolor='forestgreen', alpha=alpha,
                                      label='VPA', edgecolor='forestgreen')
                only_mvpa = (df2_h['hyp_mvpa']) & (~df2_h['hyp_vpa'])
                ax1[idx].fill_between(df2_h.index, 0, maxy, where=only_mvpa, facecolor='palegreen', alpha=alpha,
                                      label='MVPA', edgecolor='palegreen')
                only_lpa = (df2_h['hyp_lpa']) & (~df2_h['hyp_mvpa']) & (~df2_h['hyp_vpa'])
                ax1[idx].fill_between(df2_h.index, 0, maxy, where=only_lpa, facecolor='honeydew', alpha=alpha,
                                      label='LPA', edgecolor='honeydew')
                ax1[idx].fill_between(df2_h.index, 0, maxy, where=df2_h['hyp_sed'], facecolor='palegoldenrod',
                                      alpha=alpha,
                                      label='sedentary', edgecolor='palegoldenrod')

            if "sleep" in signal_categories:
                facecolors = ['royalblue', 'green', 'orange']
                # maxy = 10000
                maxy = 100
                endy = 0
                addition = (maxy / len(facecolors))
                for i, sleep_col in enumerate(sleep_cols):
                    starty = endy
                    endy = endy + addition
                    sleeping = df2_h[sleep_col]  # TODO: get a method instead of an attribute
                    ax1[idx].fill_between(df2_h.index, starty, endy, where=sleeping, facecolor=facecolors[i],
                                          alpha=alpha, label=sleep_col)
                    # ax1[idx].fill_between(df2_h.index, starty, endy, where=~sleeping, facecolor="gray", alpha=0.7,
                    #                      label=" ")
                    # ax1[idx].fill_between(df2_h.index, 0, (df2_h['wake_window_0.4']) * 200, facecolor='cyan', alpha=1,
                    #                   label='wake')

            if "diary" in signal_categories and wearable.diary_onset in df2_h.keys() and wearable.diary_offset in df2_h.keys():
                diary_event = df2_h[(df2_h[wearable.diary_onset] == True) | (df2_h[wearable.diary_offset] == True)].index
                ax1[idx].vlines(x=diary_event, ymin=0, ymax=100, facecolor='black', alpha=alpha, label='Diary',
                                linestyles="dashed")

            for i, col in enumerate(other_signals):
                #colors = ["orange", "violet", "pink", "gray"] # Change to paramters
                ax1[idx].plot(df2_h.index, df2_h[col], label=col, linewidth=1, color=colors[i], alpha=alpha)

            #maxy = 1000
            endy = 0
            addition = (maxy / len(signal_as_area))
            for i, col in enumerate(signal_as_area):
                #colors = ["orange", "violet", "pink", "gray"]
                starty = endy
                endy = endy + addition

                ax1[idx].fill_between(df2_h.index, starty, endy, where=df2_h[col], facecolor=colors[i],
                                      alpha=alpha, label=col)



            ax1[idx].set_ylabel("%d-%s\n%s" % (dfs_per_day[idx].index[-1].day,
                                               calendar.month_name[dfs_per_day[idx].index[-1].month][:3],
                                               calendar.day_name[dfs_per_day[idx].index[-1].dayofweek]),
                                rotation=0, horizontalalignment="right", verticalalignment="center")

            ax1[idx].set_xticks([])
            ax1[idx].set_yticks([])
            # ax1[idx].set_label(["Experiment Day %d" % (idx)])

            # create a twin of the axis that shares the x-axis
            if "hr" in signal_categories:
                ax2 = ax1[idx].twinx()
                # ax2.set_ylabel('mean_HR')  # we already handled the x-label with ax1
                ax2.plot(df2_h.index, df2_h[wearable.get_hr_col()], label='HR', color='red')
                ax2.set_ylim(df2_h[wearable.get_hr_col()].min() - 5, df2_h[wearable.get_hr_col()].max() + 5)
                ax2.set_xticks([])
                ax2.set_yticks([])

        ax1[0].set_title("PID = %s" % wearable.get_pid(), fontsize=16)
        ax1[-1].set_xlabel('Time')
        ax1[-1].xaxis.set_minor_locator(dates.HourLocator(interval=4))  # every 4 hours
        ax1[-1].xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes

        handles, labels = ax1[-1].get_legend_handles_labels()
        # handles2, labels2 = ax2.get_legend_handles_labels()
        # fig.legend(handles + handles2, labels + labels2, loc='lower center', ncol=4)
        # return fig
        # ax.figure.savefig('%s_signals.pdf' % (self.get_pid()))
        # fig.suptitle("%s" % self.get_pid(), fontsize=16)

        fig.legend(handles, labels, loc='lower center', ncol=len(cols), fontsize=14, shadow=True)
        fig.savefig('%s_signals.pdf' % (wearable.get_pid()), dpi=300, transparent=True, bbox_inches='tight')

