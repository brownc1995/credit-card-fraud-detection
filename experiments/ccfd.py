import argparse

from ccfd.data import *
from ccfd.model import *

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
        epochs: int,
) -> None:
    """
    Main function for experiments
    :param epochs: int, number of epochs to run over
    :return: None
    """
    logging.info('Grabbing data')
    ccfd_data = get_data()

    logging.info('Enriching and cleaning data')
    ccfd_data['log_amount'] = np.log(ccfd_data.amount + 0.0001)

    logging.info('Splitting data into training, validation, and test sets')
    train_data, train_target, val_data, val_target, test_data, test_target = train_val_test_split(ccfd_data)

    logging.info('Scaling the input data to improve performance of neural network')
    train_data, val_data, test_data = scale_data(train_data, val_data, test_data)

    logging.info('Building neural network model')
    model = build_model(train_data.shape[-1])

    logging.info('Fitting model')
    _ = fit_model(model, train_data, train_target, val_data, val_target, epochs)

    logging.info('Model has been fitted')

    logging.info('Exiting')

    return None


def _arg_parser() -> argparse.Namespace:
    """
    Argument parser
    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Main entry point for CCFD model')

    parser.add_argument(
        '--epochs',
        type=int,
        help='number of epochs to train model over'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = _arg_parser()

    main(epochs=args.epochs)
