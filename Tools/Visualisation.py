import math


class IndividualRecord:
    def __init__(self):

        self.gen_count = 0
        self.gens = []

    def add_gen(self, gen):
        self.gens.append([])
        for ind in gen:
            self.gens[self.gen_count].append(
                tuple(ind.block_architecture.evaluation_values) + tuple(ind.fitness.values))
        self.gen_count += 1

    def save(self, gen_interval, test_name, title="Fig_None", comment=None):
        import matplotlib.pyplot as plt
        import math

        plot_rows = math.ceil(len(self.gens) / gen_interval / 2)
        fig, axes = plt.subplots(
            plot_rows, 2, sharey=True, sharex=True, figsize=(20, 10 * plot_rows)
        )
        if title:
            if comment:
                title = title + "_{}".format(comment)
            fig.suptitle(title)
        for i in range(0, self.gen_count, gen_interval):
            try:
                subplot_num = i // gen_interval
                sx = subplot_num // 2
                sy = subplot_num % 2
                datax, datay, goal = map(list, zip(*self.gens[i]))
                axes[sx, sy].scatter(datax, datay)
                axes[sx, sy].set_title("Gen {}, count: {}".format(i, len(self.gens[i])))
                axes[sx, sy].set(xlabel="Param Count", ylabel="Accuracy")
            except Exception as e:
                pass

        from pathlib import Path

        path = "Output/{}/Figures".format(test_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig("Output/{}/Figures/{}".format(test_name, title))

    def goals(self, gen_interval, test_name):

        import matplotlib.pyplot as plt
        import math

        plot_cols = math.ceil(len(self.gens) / gen_interval / 2)
        fig, axes = plt.subplots(plot_cols, 2, sharey=True)
        fig.tight_layout(h_pad=2)
        fig.set_size_inches(20, 8 * plot_cols)

        goals = []
        from statistics import mean

        for i in range(0, self.gen_count, gen_interval):
            try:
                datax, datay, goal = map(list, zip(*self.gens[i]))

                goals += [(i, mean(g)) for g in goal]
            except Exception as e:
                pass

        import matplotlib.figure
        import matplotlib.backends.backend_agg as agg

        fig = matplotlib.figure.Figure(figsize=(45, 15))
        agg.FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)

        ax.title.set_text("Goals")

        ax.scatter(
            [ind[0] for ind in goals],
            [ind[1] for ind in goals],
            facecolor=(0.7, 0.7, 0.7),
            zorder=-1,
        )

        from pathlib import Path

        path = "Output/{}/Figures".format(test_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig("Output/{}/Figures/goals".format(test_name))

    def pareto(self, test_name):
        from Demos import get_global

        filter_funcs = get_global("filter_function_args")

        individuals = [list(ind) for ind in self.gens[-1]]
        best_models = [
            [param_count, 0]
            for param_count in list(set([ind[0] for ind in individuals]))
        ]
        best_models.sort(key=lambda x: x[0])

        for ind in individuals:
            bm_index = [pcount for (pcount, acc) in best_models].index(ind[0])
            if ind[1] > best_models[bm_index][1]:
                best_models[bm_index][1] = ind[1]

        pareto_inds = [best_models[0]]

        for ind_to_compare in best_models[1:]:

            is_dominated = False

            for existing_ind in pareto_inds:

                if a_dominates_b(existing_ind, ind_to_compare) or (
                    set(ind_to_compare) == set(existing_ind)
                ):

                    is_dominated = True
                    break

                elif a_dominates_b(ind_to_compare, existing_ind):
                    pareto_inds.remove(existing_ind)

            if is_dominated:
                continue
            else:
                pareto_inds.append(ind_to_compare)

        pareto_x = [ind[0] for ind in pareto_inds]
        pareto_y = [ind[1] for ind in pareto_inds]

        import matplotlib.backends.backend_agg as agg
        import matplotlib.figure

        fig = matplotlib.figure.Figure(figsize=(45, 15))
        agg.FigureCanvasAgg(fig)

        ax = fig.add_subplot(1, 3, 1)
        ax.title.set_text("Population")
        ax.set_ylim(bottom=0, top=100)
        ax.scatter(
            [ind[0] for ind in individuals],
            [ind[1] for ind in individuals],
            facecolor=(0.7, 0.7, 0.7),
            zorder=-1,
        )
        m = -filter_funcs[1][1] / filter_funcs[1][0]
        c = filter_funcs[0][1] - (m * filter_funcs[0][0])
        y_point = [0, c]
        x_point = [-c / m, 0]
        ax.plot(filter_funcs[0][0], filter_funcs[0][1], "go")
        ax.plot(x_point, y_point)

        ax = fig.add_subplot(1, 3, 2)
        ax.set_xscale("log")
        ax.title.set_text("Best for Each Param Count")
        ax.set_ylim(bottom=0, top=100)
        ax.scatter(
            [ind[0] for ind in best_models],
            [ind[1] for ind in best_models],
            facecolor=(0.7, 0.7, 0.7),
            zorder=-1,
        )

        ax = fig.add_subplot(1, 3, 3)
        ax.plot(pareto_x, pareto_y)
        ax.scatter(pareto_x, pareto_y, facecolor=(0.7, 0.7, 0.7), zorder=-1)
        # ax.set_xscale("log")
        ax.title.set_text("Pareto Front")
        ax.set_ylim(bottom=0, top=100)

        from pathlib import Path

        path = "Output/{}/Figures".format(test_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig("Output/{}/Figures/pareto".format(test_name))

        return pareto_inds


def a_dominates_b(a, b):

    n_better = 0

    # First index is parameter count, thus we want a[0] < b[0]
    if a[0] < b[0]:
        n_better += 1

    # Second index is accuracy, thus we want a[1] > b[1]
    if a[1] > b[1]:
        n_better += 1

    if n_better == 2:
        return True

    return False


def plot_hof_pareto(hof, test_name):
    import matplotlib

    x = [i.block_architecture.evaluation_values[0] for i in hof.items]
    y = [i.block_architecture.evaluation_values[1] for i in hof.items]
    z = [i.block_architecture.evaluation_values[2] for i in hof.items]
    import matplotlib.backends.backend_agg as agg

    fig = matplotlib.figure.Figure(figsize=(15, 15))
    agg.FigureCanvasAgg(fig)

    padding = 1.1
    max_x = max(x) * padding
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(x, y, facecolor=(0.7, 0.7, 0.7), zorder=-1)
    ax.xscale = "log"

    for item in [(x[i], y[i]) for i in range(1, len(x))]:
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (item[0], 0),
                max_x - item[0],
                item[1],
                lw=0,
                facecolor=(1.0, 0.8, 0.8),
                zorder=-10,
            )
        )

    ax.set_xscale("log")
    ax.set_ylim(bottom=0, top=100)

    from pathlib import Path

    path = "Output/{}/Figures".format(test_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig("Output/{}/Figures/hof_pareto".format(test_name))
