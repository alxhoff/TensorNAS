import matplotlib.figure


class IndividualRecord:
    def __init__(self):

        self.gen_count = 0
        self.gens = []

    def add_gen(self, gen):
        self.gens.append([])
        for ind in gen:
            self.gens[self.gen_count].append(
                (ind.block_architecture.param_count, ind.block_architecture.accuracy)
            )
        self.gen_count += 1

    def save(self, gen_interval, test_name, title="Fig_None", comment=None):
        import matplotlib.pyplot as plt
        import math

        plot_cols = math.ceil(len(self.gens) / gen_interval / 2)
        fig, axes = plt.subplots(plot_cols, 2, sharey=True)
        fig.tight_layout(h_pad=2)
        fig.set_size_inches(20, 10 * plot_cols)
        if title:
            if comment:
                title = title + "_{}".format(comment)
            fig.suptitle(title)
        for i in range(0, self.gen_count, gen_interval):
            try:
                subplot_num = i // gen_interval
                sx = subplot_num // 2
                sy = subplot_num % 2
                datax, datay = map(list, zip(*self.gens[i]))
                axes[sx, sy].scatter(datax, datay)
                axes[sx, sy].set_title("Gen {}, count: {}".format(i, len(self.gens[i])))
                axes[sx, sy].set(xlabel="Param Count", ylabel="Accuracy")
            except Exception as e:
                pass

        from pathlib import Path

        path = "Output/{}/Figures".format(test_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig("Output/{}/Figures/{}".format(test_name, title))

    def pareto(self, test_name):
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

        skyline = [best_models[0]]

        for ind in best_models[1:]:

            is_dominated = False

            for s_ind in skyline:

                if a_dominates_b(s_ind, ind, [0], [1]):

                    is_dominated = True
                    break

                elif a_dominates_b(ind, s_ind, [0], [1]):
                    skyline.remove(s_ind)

            if is_dominated:
                continue
            else:
                skyline.append(ind)

        x = [ind[0] for ind in skyline]
        y = [ind[1] for ind in skyline]

        import matplotlib.backends.backend_agg as agg

        fig = matplotlib.figure.Figure(figsize=(45, 15))
        agg.FigureCanvasAgg(fig)

        ax = fig.add_subplot(1, 3, 1)
        ax.set_xscale("log")
        ax.title.set_text("Population")
        ax.set_ylim(bottom=0, top=100)
        ax.scatter(
            [ind[0] for ind in individuals],
            [ind[1] for ind in individuals],
            facecolor=(0.7, 0.7, 0.7),
            zorder=-1,
        )
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
        ax.plot(x, y)
        ax.scatter(x, y, facecolor=(0.7, 0.7, 0.7), zorder=-1)
        ax.set_xscale("log")
        ax.title.set_text("Pareto Front")
        ax.set_ylim(bottom=0, top=100)

        from pathlib import Path

        path = "Output/{}/Figures".format(test_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig("Output/{}/Figures/pareto".format(test_name))


def a_dominates_b(a, b, to_min, to_max):

    n_better = 0

    for f in to_min:
        if a[f] > b[f]:
            return False
        n_better += a[f] < b[f]

    for f in to_max:
        if a[f] < b[f]:
            return False
        n_better += a[f] > b[f]

    if n_better > 0:
        return True
    return False


def plot_hof_pareto(hof, test_name):
    import matplotlib

    x = [i.block_architecture.param_count for i in hof.items]
    y = [i.block_architecture.accuracy for i in hof.items]

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
    fig.savefig("Output/{}/Figures/pareto".format(test_name))
