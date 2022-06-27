class LatexLayer:
    def __init__(self, name, string):
        self.name = name
        self.string = string


class LatexWriter:
    def __init__(self):
        self.conv_count = 0
        self.pool_count = 0
        self.dense_count = 0
        self.flatten_count = 0

        import os

        dir = os.path.join(os.getcwd())

        if not os.path.exists(os.path.join(os.getcwd(), "PlotNeuralNet")):
            import git

            git.Git(dir).clone("https://github.com/HarisIqbal88/PlotNeuralNet.git")
            print("PlotNeuralNet cloned")

        mod_name = "PlotNeuralNet.pycore.tikzeng"
        self.mod = __import__(mod_name, globals(), locals(), ["*"])
        print("tikzeng imported")

    def _recurse_blocks(self, blocks):
        from TensorNAS.Core.LayerBlock import LayerBlock

        ret = []

        if len(blocks) == 1:
            if issubclass(blocks[0].__class__, LayerBlock):
                return blocks

        for bl in blocks:
            try:
                ret.extend(
                    self._recurse_blocks(bl.input_blocks)
                    + self._recurse_blocks(bl.middle_blocks)
                    + self._recurse_blocks(bl.output_blocks)
                )
            except Exception as e:
                break
        return ret

    def _get_flattened_model(self, model):
        return self._recurse_blocks(
            model.input_blocks + model.middle_blocks + model.output_blocks
        )

    def _get_latex_layer(self, layer, to):
        from TensorNAS.Layers import SupportedLayers

        print("wait here")
        if layer.layer_type == SupportedLayers.CONV2D:
            from TensorNAS.Layers.Conv2D import Args

            self.conv_count += 1
            name = "conv{}".format(self.conv_count)
            string = self.mod.to_Conv(
                name,
                layer.layer.args[Args.FILTERS],
                layer.layer.inputshape.get()[-1],
                offset="(0,0,0)",
                to=to,
                height=layer.layer.inputshape.get()[1],
                depth=layer.layer.inputshape.get()[1],
                width=layer.layer.outputshape.get()[-1],
            )
            return LatexLayer(name, string)
        if layer.layer_type == SupportedLayers.MAX_POOL2D:
            self.pool_count += 1
            name = "pool{}".format(self.pool_count)
            string = self.mod.to_Pool(
                name, offset="(0,0,0)", to=to, caption="MaxPool2D"
            )
            return LatexLayer(name, string)
        if layer.layer_type == SupportedLayers.OUTPUTDENSE:
            from TensorNAS.Layers.Dense import Args

            self.dense_count += 1
            name = "dense{}".format(self.dense_count)
            string = self.mod.to_SoftMax(
                name, layer.layer.args[Args.UNITS], "(3,0,0)", to
            )
            return LatexLayer(name, string)
        if layer.layer_type == SupportedLayers.FLATTEN:
            self.flatten_count += 1
            name = "flatten{}".format(self.flatten_count)
            string = self.mod.to_Conv(
                name,
                layer.layer.outputshape.get()[-1],
                1,
                offset="(0,0,0)",
                to=to,
                height=layer.layer.outputshape.get()[-1],
                depth=1,
                width=1,
            )
            return LatexLayer(name, string)

    def create_arch(self, model):

        fm = self._get_flattened_model(model)
        latex_arch = [self.mod.to_head(".."), self.mod.to_cor(), self.mod.to_begin()]

        to_name = "(0,0,0)"

        for layer in fm:
            ll = self._get_latex_layer(layer, to_name)
            try:
                latex_arch.append(ll.string)
            except Exception:
                print("wait here")
            to_name = "({}-east)".format(ll.name)
            if len(latex_arch) == 2:
                break

        latex_arch.append(self.mod.to_end())

        # latex_arch = [
        #     self.mod.to_head('..'),
        #     self.mod.to_cor(),
        #     self.mod.to_begin(),
        #     self.mod.to_Conv("conv1", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2),
        #     self.mod.to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
        #     self.mod.to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2),
        #     self.mod.to_connection("pool1", "conv2"),
        #     self.mod.to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
        #     self.mod.to_SoftMax("soft1", 10, "(3,0,0)", "(pool1-east)", caption="SOFT"),
        #     self.mod.to_connection("pool2", "soft1"),
        #     self.mod.to_end()
        # ]

        self.compile_arch(latex_arch)

        return latex_arch

    def compile_arch(self, arch):
        import os

        try:
            os.mkdir(os.path.join(os.getcwd(), "PlotNeuralNet/results"))
        except Exception:
            pass
        self.mod.to_generate(arch, "PlotNeuralNet/results/test.tex")

        os.system("cd PlotNeuralNet/results && bash ../tikzmake.sh test")
