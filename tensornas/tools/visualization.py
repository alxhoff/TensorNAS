import os
class IndividualRecord:
    def __init__(self):

        self.gen_count = 0
        self.gens = []

    def add_gen(self, gen):
        self.gens.append([])
        for ind in gen:
            self.gens[self.gen_count].append(ind.fitness.values)
        self.gen_count += 1

    def show(self, gen_interval):
        import matplotlib.pyplot as plt
        import math

        plot_cols = math.ceil(len(self.gens) / gen_interval / 2)
        fig, axes = plt.subplots(2, plot_cols, sharex=True, sharey=True)
        fig.text(0.65, 0.04, 'Param Count')
        fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')
        for i in range(1, self.gen_count + 1, gen_interval):
            try:
                for j in range(2):
                    subplot_num = i // gen_interval
                    sy = subplot_num // 2
                    sx = subplot_num % 2
                    datax, datay = map(list, zip(*self.gens[i - 1]))
                    axes[sx, sy].scatter(datax, datay)
                    #axes[sx, sy].set_title("Gen {}".format(i))

                    axes[sx, sy].label_outer()
            except Exception as e:
                pass
        #axes[sx, sy].set(xlabel="Param Count", ylabel="Accuracy")
        folder='C:\\Users\\mehta\\Desktop\\TensorNAS\\Results'
        i=0
        while 1:
            if os.path.isfile(f'{folder}\\Generations_Visulaization_{i}.eps'):
                i+=1
            else:
                plt.savefig(f'{folder}\\Generations_Visulaization_{i}.eps', format='eps')
                break

        #plt.savefig(f'{filename}{i}.eps', format='eps')



        plt.show()
