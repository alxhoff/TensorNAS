from TensorNAS.Layers.Conv2D import Layer


class Layer(Layer):
    def get_keras_layers(self, input_tensor):
        import tensorflow as tf
        from TensorNAS.Core.Layer import gen_regularizer

        return tf.keras.layers.Conv2D(
            filters=self.args.get(self.get_args_enum().FILTERS),
            kernel_size=self.args.get(self.get_args_enum().KERNEL_SIZE),
            strides=self.args.get(self.get_args_enum().STRIDES),
            input_shape=self.inputshape.get(),
            activation=self.args.get(self.get_args_enum().ACTIVATION).value(),
            padding=self.args.get(self.get_args_enum().PADDING).value(),
            dilation_rate=self.args.get(self.get_args_enum().DILATION_RATE),
            kernel_regularizer=gen_regularizer(self.args.get(self.get_args_enum().REGULARIZER)),
            kernel_initializer=self.args.get(self.get_args_enum().INITIALIZER).value(),
        )(input_tensor)

if __name__ == "__main__":
    """
    """ 
    from TensorNAS.Blocks.BlockArchitectures.MNISTBlockArchitecture import Block as MNIST_BlArch
    from TensorNAS.Blocks.SubBlocks.FeatureExtractionBlock import Block as FeatureExtractionBlock
    from TensorNAS.Layers.Conv2D import Args, ArgActivations, ArgPadding, ArgRegularizers, ArgInitializers
   
    args={
        "FILTERS" : 67,
        "KERNEL_SIZE" : (3,3),
        "STRIDES" : (1,1),
        "PADDING" : ArgPadding.SAME,
        "DILATION_RATE" : (1,1),
        "ACTIVATION":ArgActivations.NONE,
        "REGULARIZER":(ArgRegularizers.NONE,0),
        "INITIALIZER":ArgInitializers.GLOROT_UNIFORM,
        }

    MNIST_BlArch_instance=MNIST_BlArch(input_shape=(28,28,1),batch_size=512,test_batch_size=32, optimizer='adam',class_count=10) # outputshape = (1,10)
    feat_extract_block_instance = FeatureExtractionBlock(input_shape=(28,28,1), parent_block=MNIST_BlArch_instance, args=None) # outputshape = (28,28,67)
    conv_2d_instance = Layer(input_shape=(28,28,1),parent_block=feat_extract_block_instance, args=args) #core.layer.layershape (28,28,67)
    
    method_names = [name for name in dir(conv_2d_instance) if callable(getattr(conv_2d_instance, name))]
    for item in method_names:
        print(item)
    """
    self.mutation_funcs = [
                func
                for func in dir(self)
                if callable(getattr(self, func))
                and re.search(r"^_mutate(?!_self)", func)
            ]
    """
    pass