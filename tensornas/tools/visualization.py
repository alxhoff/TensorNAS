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

    def show(self, gen_interval, title=None):
        import matplotlib.pyplot as plt
        import math

        plt.ion()
        plt.show()
        plot_cols = math.ceil(len(self.gens) / gen_interval / 2)
        fig, axes = plt.subplots(plot_cols, 2, sharey=True)
        fig.tight_layout(h_pad=2)
        if title:
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
                axes[sx, sy].label_outer()
            except Exception as e:
                pass

        plt.draw()
        plt.pause(0.001)


def plot_hof_pareto(hof, filename="pareto"):
    import matplotlib
    import matplotlib.pyplot as plt

    x = [i.block_architecture.param_count for i in hof.items]
    y = [i.block_architecture.accuracy for i in hof.items]

    import matplotlib.backends.backend_agg as agg

    fig = matplotlib.figure.Figure(figsize=(15, 15))
    agg.FigureCanvasAgg(fig)
    plt.figure(figsize=(15, 5))

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

    fig.savefig("pareto_" + filename)
