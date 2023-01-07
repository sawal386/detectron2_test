from detectron2.engine import DefaultTrainer

class ModelFineTuner:

    def __init__(self, model, train_data_name, batch_size, im_per_batch,
                 n_class, output_path):
        """
        :param (BaseModel) model: the model used for the task-in-hand
        :param (str) train_data_name: the name of the training data
        :param (int) batch_size: the batch size
        :param (int) im_per_batch: number of images per_batch
        :param (int) n_class: number of foreground classes in the image
        :param (str) output_path: location where the output is saved

        """

        self.model = model
        self.model.setup_training(train_data_name, im_per_batch, batch_size,
                       n_class, output_path)


    def run_training(self, learning_rate, n_iterations):
        """
        trains the model according to the specified hyperparameters
        :param (float) learning_rate: the learning rate to be used
        :param (int) n_iterations: the total number of training iterations
        """

        self.model.assign_hyperparameters(learning_rate, n_iterations)
        print("info:", self.model.cfg.SOLVER.BASE_LR, self.model.cfg.SOLVER.MAX_ITER)
        trainer = DefaultTrainer(self.model.get_model_cfg())
        trainer.resume_or_load(resume=False)
        trainer.train()





