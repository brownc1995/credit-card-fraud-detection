import argparse

from ccfd.data import *
from ccfd.model import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
        epochs: int,
) -> None:
    """
    Main function for experiments
    :param epochs: int, number of epochs to run over
    :return: None
    """
    #############################################
    #
    # setup
    #
    #############################################

    logging.info('Grabbing data')
    ccfd_data = get_data()

    logging.info('Splitting data into training, validation, and test sets')
    train_data, train_target, val_data, val_target, test_data, test_target = train_val_test_split(ccfd_data)

    logging.info('Scaling the input data to improve performance of neural network')
    train_data, val_data, test_data = scale_data(train_data, val_data, test_data)

    logging.info('Creating tf.data.Datasets')
    train_dataset, val_dataset, test_dataset = make_all_datasets(
        train_data,
        train_target,
        val_data,
        val_target,
        test_data,
        test_target
    )

    logging.info('Setting up neural network')
    initial_bias = calc_initial_bias(ccfd_data)
    input_shape = train_data.shape[-1]

    #############################################
    #
    # vanilla network
    #
    #############################################

    logging.info('Creating vanilla model')
    model = build_model(
        input_shape=input_shape,
        output_bias=initial_bias
    )

    logging.info('Fitting vanilla model')
    _ = fit_model(
        model,
        train_dataset,
        val_dataset,
        epochs=epochs,
    )

    logging.info('Evaluating model performance')
    baseline_results = model.evaluate(
        test_dataset,
        verbose=0
    )
    test_predictions_baseline = model.predict(test_dataset)
    log_model_performance(model, baseline_results, test_target, test_predictions_baseline)

    #############################################
    #
    # class-weighted network
    #
    #############################################

    logging.info('Creating class-weighted model')
    class_weight = set_class_weights(ccfd_data)
    model_cw = build_model(
        input_shape=input_shape,
        output_bias=initial_bias
    )

    logging.info('Fitting class-weighted model')
    _ = fit_model(
        model_cw,
        train_dataset,
        val_dataset,
        epochs=epochs,
        class_weight=class_weight
    )

    logging.info('Evaluating class-weighted model performance')
    cw_results = model_cw.evaluate(
        test_dataset,
        verbose=0
    )
    test_predictions_cw = model_cw.predict(test_dataset)
    log_model_performance(model_cw, cw_results, test_target, test_predictions_cw)

    #############################################
    #
    # resampled network
    #
    #############################################

    logging.info('Creating model oversampling from positive transactions')
    logging.info('Resampling the data')
    pos_data, pos_target, neg_data, neg_target = pos_neg_data(train_data, train_target)
    pos_dataset, neg_dataset = make_datasets_pos_neg(pos_data, pos_target, neg_data, neg_target)
    resampled_dataset = resample_dataset(pos_dataset, neg_dataset)
    resampled_steps_per_epoch = resample_steps_per_epoch(ccfd_data)

    logging.info('Creating resampled model')
    model_rs = build_model(
        input_shape=input_shape,
        output_bias=initial_bias
    )

    logging.info('Fitting the resampled model')
    _ = fit_model(
        model_rs,
        resampled_dataset,
        val_dataset,
        steps_per_epoch=resampled_steps_per_epoch,
        epochs=epochs,
        resampled=True
    )

    logging.info('Evaluating resampled model performance')
    rs_results = model_rs.evaluate(
        test_dataset,
        verbose=0
    )
    test_predictions_rs = model_rs.predict(test_dataset)
    log_model_performance(model_rs, rs_results, test_target, test_predictions_rs)

    logging.info('Experiments ran successfully')
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
