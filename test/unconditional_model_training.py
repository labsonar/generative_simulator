from argparse import ArgumentParser
from torch.cuda import device_count

import model_lib.dataset as model_dataset
import model_lib.model as model_model
import model_lib.learner as model_learner

def main(args):

    training_config = model_learner.TrainingConfig.from_argparse(args)
    model_config = model_model.UnconditionalConfig.from_argparse(args)

    dataset = model_dataset.FolderDataset(
                    base_dir=args.data_dirs,
                    processing=model_dataset.SplitWindow(
                                    args.n_samples,
                                    args.overlap))

    model_learner.train(
            override=args.override,
            backup=not args.do_not_backup,
            dataset=dataset,
            model_config=model_config,
            training_config=training_config)

if __name__ == '__main__':
    parser = ArgumentParser(description='train (or resume training) a DiffWave model')
    parser.add_argument('data_dirs', help='directory from which to read .wav files for training')
    parser.add_argument('-n','--n_samples', default=22050*6, type=int,
        help='number of samples')
    parser.add_argument('-o','--overlap', default=22050*3, type=int,
        help='number of overlap samples')
    parser.add_argument('--override', default=False, type=bool,
        help='override model')
    parser.add_argument('--do_not_backup', default=True, type=bool,
        help='do not make the backup model when overriding')

    model_learner.TrainingConfig.add_arg_opt(parser)
    model_model.UnconditionalConfig.add_arg_opt(parser)

    main(parser.parse_args())