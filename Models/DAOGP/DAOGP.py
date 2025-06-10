from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector import ConceptDriftDetector
from ConceptDriftManager.ConceptDriftHandler import ConceptDriftHandler
from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData
from Kernels.KernelsPoolManager import KernelPoolHandler
from sklearn.model_selection import train_test_split
from Utils import Util, Printer
from sklearn.metrics import mean_squared_error, r2_score
from Kernels.KernelsHyperparams import KernelHyperparameterOptimizer
from InducingPoints.InducingPoints import InducingPoints
from PerformanceRecorder.MBPerformance import  MBPerformance
from PerformanceRecorder.MBPerformanceManager import MBPerformanceManager
from Plotter import Plotter
import numpy as np


def dao_gp(X_train, y_train, INITIAL_BATCH_SIZE, INCREMENT_SIZE, DECAY_GAMMA, MAX_INDUCING, INITIAL_KERNEL, KPI, Z, SAFE_AREA_THRESHOLD,
           KERNEL_POOL, UNCERTAINTY_THRESHOLD, STRETCH_FACTOR=None):
    n_samples, n_features = X_train.shape
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()

    mbPerformanceManagerTr = MBPerformanceManager()
    mbPerformanceManagerVl = MBPerformanceManager()
    time_counter = 0


    X_base = X_train[:INITIAL_BATCH_SIZE]
    y_base = y_train[:INITIAL_BATCH_SIZE]

    # VALIDATION_FRACTION = .5
    # if len(X_base) == 1:
    #     X_base_tr = X_base_vl = X_base
    #     y_base_tr = y_base_vl = y_base
    # else:
    #     X_base_tr, X_base_vl, y_base_tr, y_base_vl = train_test_split(X_base, y_base, test_size=VALIDATION_FRACTION)
    X_base_tr = X_base_vl = X_base
    y_base_tr = y_base_vl = y_base
    X_base_time = Util.append_timestamps([], len(X_base_tr), time_counter)
    # print(X_base_time)
    kernel, kernel_params, opt_dict, kernel_func, best_kernel_args, best_r2, best_mse, best_r2_tr, best_mse_tr = \
        KernelPoolHandler.pick_best_kernel(INITIAL_KERNEL, X_base_tr, y_base_tr, X_base_vl, y_base_vl, KERNEL_POOL, n_features, STRETCH_FACTOR, KPI, tol=1e-6)



    noise = opt_dict["noise"]
    kernel_args = {k: v for k, v in opt_dict.items() if k != "noise"}
    K = kernel_func(X_base_tr, X_base_tr, **kernel_args) + noise * np.eye(len(y_base_tr))
    K_inv_undecayed = np.linalg.inv(K)

    # Statistical Meta-Data Saved about First Mini-Batch (BaseModel)
    miniBatchMetaData = MiniBatchMetaData(best_r2, best_mse)
    memoryManager.add_mini_batch_data(miniBatchMetaData)

    # PerformanceRecorded for Trn and Vl
    mbPerformanceVl = MBPerformance(INITIAL_BATCH_SIZE,best_r2, best_mse)
    mbPerformanceManagerVl.add_mini_batch_data(mbPerformanceVl)
    mbPerformanceTr = MBPerformance(INITIAL_BATCH_SIZE,best_r2_tr, best_mse_tr)
    mbPerformanceManagerTr.add_mini_batch_data(mbPerformanceTr)


    for i in range(INITIAL_BATCH_SIZE, n_samples, INCREMENT_SIZE):
        time_counter+=1
        if time_counter > 1:
            K_inv_undecayed = K_inv_decayed
        print("**********************************ITERATION ", i, "***********************************")

        X_inc = X_train[i: i + INCREMENT_SIZE]
        y_inc = y_train[i: i + INCREMENT_SIZE]

        # if len(X_inc) == 1:
        #     X_inc_tr = X_inc_vl = X_inc
        #     y_inc_tr = y_inc_vl = y_inc
        # else:
        #     X_inc_tr, X_inc_vl, y_inc_tr, y_inc_vl = train_test_split(X_inc, y_inc, test_size=VALIDATION_FRACTION)
        X_inc_tr = X_inc_vl = X_inc
        y_inc_tr = y_inc_vl = y_inc
        for x_new, y_new in zip(X_inc_tr, y_inc_tr):
            predictive_var = Util.compute_predictive_variance(X_window=X_base_tr, y_window=y_base_tr, X_base_time=X_base_time, gamma=DECAY_GAMMA,
                                                              K_inv_undecayed=K_inv_undecayed,
                                                              x_new=x_new, noise=noise,
                                                              kernel_func=kernel_func, **kernel_args)
            if predictive_var > UNCERTAINTY_THRESHOLD:
                x_new = np.atleast_2d(x_new)
                # Attempt Woodbury update.
                K_inv_updated = Util.woodbury_update(K_inv_undecayed, X_base_tr, x_new, kernel_func, kernel_args, noise)
                if K_inv_updated is not None: # is woodbury stable
                    K_inv_undecayed = K_inv_updated
                    X_base_tr = np.vstack([X_base_tr, x_new])
                    y_base_tr = np.append(y_base_tr, y_new)
                    X_base_time = Util.append_timestamps(X_base_time, len(x_new), time_counter)
                else:
                    X_base_tr = np.vstack([X_base_tr, x_new])
                    y_base_tr = np.append(y_base_tr, y_new)
                    X_base_time = Util.append_timestamps(X_base_time, len(x_new), time_counter)
                    K = kernel_func(X_base_tr, X_base_tr, **kernel_args) + noise  * np.eye(len(y_base_tr))
                    K_inv_undecayed = np.linalg.inv(K)



        min_window_length = 7
        # if len(memoryManager.mini_batch_data) < min_window_length:
        #     K, K_inv_undecayed = Util.optimizeKernelHyperparamsGeneric(X_base_tr, y_base_tr, kernel, kernel_params, kernel_func)

        # Predict and Evaluate on 1. Train 2. Validation #####################
        mu_train, _ = Util.computeGP(X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv_undecayed,
                                         X_star=X_base_tr, noise=noise,
                                         kernel_func=kernel_func, **kernel_args)

        # mu_train, _ = Util.compute_weighted_GP(X_train=X_base_tr, y_train=y_base_tr, X_train_time=X_base_time, gamma=DECAY_GAMMA, X_star=X_base_tr, noise=noise, kernel_func=kernel_func, **kernel_args)
        # put new one with optimization
        mse_train = mean_squared_error(y_base_tr, mu_train)
        r2_train = r2_score(y_base_tr, mu_train)
        print(f"Update {i}: TRAIN MSE={mse_train:.4f}, R^2={r2_train:.4f}")

        # Also evaluate on the Validation Set.
        val_pred, _ = Util.computeGP(X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv_undecayed,
                                         X_star=X_inc_vl, noise=noise,
                                         kernel_func=kernel_func, **kernel_args)
        # val_pred, _ = Util.compute_weighted_GP(X_train=X_base_tr, y_train=y_base_tr, X_train_time=X_base_time, gamma=DECAY_GAMMA, X_star=X_inc_vl, noise=noise, kernel_func=kernel_func, **kernel_args)
        val_mse = mean_squared_error(y_inc_vl, val_pred)
        val_r2 = r2_score(y_inc_vl, val_pred)
        print("RES: Validation : ", "MSE", val_mse, "R^2", val_r2)
        ############# End Prediction and Evaluation ############################

        miniBatchMetaData = MiniBatchMetaData(val_r2, val_mse)
        memoryManager.add_mini_batch_data(miniBatchMetaData)

        # Performance Recorded
        mbPerformanceTr = MBPerformance(i + INCREMENT_SIZE,r2_train, mse_train)
        mbPerformanceManagerTr.add_mini_batch_data(mbPerformanceTr)
        mbPerformanceVl = MBPerformance(i + INCREMENT_SIZE,val_r2, val_mse)
        mbPerformanceManagerVl.add_mini_batch_data(mbPerformanceVl)

        # Concept Drift
        min_window_length = 7  # duplicated intentionally here.
        max_window_length = 31
        is_drift, drift_type = ConceptDriftHandler.evaluate_cocept_drift(conceptDriftDetector, memoryManager, KPI, Z,
                                                                         min_window_length, max_window_length,
                                                                         SAFE_AREA_THRESHOLD)
        print("is_drift", is_drift, "drift_type", drift_type)
        if is_drift:
            print('##### is_drift ', is_drift, "#####")
            optimized_dict = KernelHyperparameterOptimizer.optimize_hyperparameters(X_base_tr, y_base_tr, kernel,
                                                                                    kernel_params)
            noise = optimized_dict["noise"]
            kernel_args = {k: v for k, v in optimized_dict.items() if k != "noise"}
            K = kernel_func(X_base_tr, X_base_tr, **kernel_args) + noise * np.eye(len(y_base_tr))
            K_inv_undecayed = np.linalg.inv(K)  # Check if we can apply woodbury here to avoid the inverse?

            mu_train, _ = Util.computeGP(X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv_undecayed,
                                             X_star=X_base_tr, noise=noise,
                                             kernel_func=kernel_func, **kernel_args)
            # mu_train, _ = Util.compute_weighted_GP(X_train=X_base_tr, y_train=y_base_tr, X_train_time=X_base_time, gamma=DECAY_GAMMA, X_star=X_base_tr, noise=noise, kernel_func=kernel_func, **kernel_args)
            # put new one with optimization
            mse_train = mean_squared_error(y_base_tr, mu_train)
            r2_train = r2_score(y_base_tr, mu_train)
            print(f"Reopt at update {i}: TRAIN MSE={mse_train:.4f}, R^2={r2_train:.4f}")

            # Evaluate on validation set.
            val_pred, _ = Util.computeGP(X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv_undecayed,
                                             X_star=X_inc_vl, noise=noise,
                                             kernel_func=kernel_func, **kernel_args)
            # val_pred, _ = Util.compute_weighted_GP(X_train=X_base_tr, y_train=y_base_tr, X_train_time=X_base_time, gamma=DECAY_GAMMA, X_star=X_inc_vl, noise=noise, kernel_func=kernel_func, **kernel_args)
            val_mse = mean_squared_error(y_inc_vl, val_pred)
            val_r2 = r2_score(y_inc_vl, val_pred)
            print(f"Reopt at update {i}: Validation MSE={val_mse:.4f}, R^2={val_r2:.4f}")

            # remove latest entry
            memoryManager.remove_last_mini_batch_data()
            miniBatchMetaData = MiniBatchMetaData(val_r2, val_mse)
            memoryManager.add_mini_batch_data(miniBatchMetaData)

            # Performance Recorded
            mbPerformanceManagerTr.remove_last_mini_batch_data()
            mbPerformanceManagerVl.remove_last_mini_batch_data()
            mbPerformanceTr = MBPerformance(i + INCREMENT_SIZE,r2_train, mse_train)
            mbPerformanceManagerTr.add_mini_batch_data(mbPerformanceTr)
            mbPerformanceVl = MBPerformance(i + INCREMENT_SIZE, val_r2, val_mse)
            mbPerformanceManagerVl.add_mini_batch_data(mbPerformanceVl)

            # Reassess (update is_drift and drift_type)
            is_drift, drift_type = ConceptDriftHandler.evaluate_cocept_drift(conceptDriftDetector, memoryManager, KPI,
                                                                             Z, min_window_length, max_window_length)
            if is_drift and drift_type == "LT":
                print("*******LT*****")
                kernel, kernel_params, opt_dict, kernel_func, kernel_args, best_r2, best_mse, best_r2_tr, best_mse_tr = \
                    KernelPoolHandler.pick_best_kernel(kernel, X_base_tr, y_base_tr, X_inc_vl, y_inc_vl,
                                                       KERNEL_POOL, n_features, STRETCH_FACTOR, KPI)
                memoryManager.remove_last_mini_batch_data()
                miniBatchMetaData = MiniBatchMetaData(best_r2, best_mse)
                memoryManager.add_mini_batch_data(miniBatchMetaData)

                # PerformanceRecorded for Trn and Vl
                mbPerformanceManagerVl.remove_last_mini_batch_data()
                mbPerformanceManagerTr.remove_last_mini_batch_data()
                mbPerformanceVl = MBPerformance(i + INCREMENT_SIZE, best_r2, best_mse)
                mbPerformanceManagerVl.add_mini_batch_data(mbPerformanceVl)
                mbPerformanceTr = MBPerformance(i + INCREMENT_SIZE, best_r2_tr, best_mse_tr)
                mbPerformanceManagerTr.add_mini_batch_data(mbPerformanceTr)


        # print("before select inducing points X_base_tr", X_base_tr, "y_base_tr", y_base_tr, "X_base_time", X_base_time)

        X_base_tr, y_base_tr, K_inv_decayed, X_base_time = InducingPoints.select_inducing_points(
                                                    X_base_tr, y_base_tr, X_base_tr, noise,
                                                    DECAY_GAMMA,kernel_func, kernel_args, MAX_INDUCING, X_base_time)

        # print("after select inducing points X_base_tr", X_base_tr, "y_base_tr", y_base_tr, "X_base_time", X_base_time)

        # K_inv_decayed = Util.apply_decay_with_time(K_inv_undecayed, DECAY_GAMMA, X_base_time)

        mu_tr, _ = Util.computeGP(X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv_decayed,
                                           X_star=X_inc_tr, noise=noise,
                                           kernel_func=kernel_func, **kernel_args)
        # mu_tr, _ = Util.compute_weighted_GP(X_train= X_base_tr, y_train=y_base_tr, X_train_time=X_base_time, gamma=DECAY_GAMMA, X_star=X_inc_tr, noise=noise, kernel_func=kernel_func, ** kernel_args)

        mse_tr = mean_squared_error(y_inc_tr, mu_tr)
        r2_tr = r2_score(y_inc_tr, mu_tr)
        print(f"Update (after selecting inducing points) {i + 1} | TRAINING MSE: {mse_tr:.4f}, R^2: {r2_tr:.4f}")

        mu_vl, _ = Util.computeGP(X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv_decayed,
                                         X_star=X_inc_vl, noise=noise,
                                         kernel_func=kernel_func, **kernel_args)
        # mu_vl, _ = Util.compute_weighted_GP(X_train= X_base_tr, y_train=y_base_tr, X_train_time=X_base_time, gamma=DECAY_GAMMA, X_star=X_inc_vl, noise=noise, kernel_func=kernel_func, ** kernel_args)

        mse_val = mean_squared_error(y_inc_vl, mu_vl)
        r2_val = r2_score(y_inc_vl, mu_vl)
        print(f"Update (after selecting inducing points) {i + 1} | VALIDATION MSE: {mse_val:.4f}, R^2: {r2_val:.4f}")

    # PerformanceRecorder
    epoch_list = mbPerformanceManagerTr.get_epochs_list()
    r2_list_tr = mbPerformanceManagerTr.get_r2_list()
    mse_list_tr = mbPerformanceManagerTr.get_mse_list()
    r2_list_vl = mbPerformanceManagerVl.get_r2_list()
    mse_list_vl = mbPerformanceManagerVl.get_mse_list()
    print("Epochs")
    Printer.print_list_tabulate(epoch_list)
    print("Training")
    Printer.print_list_tabulate(r2_list_tr)
    print("Validation")
    Printer.print_list_tabulate(r2_list_vl)

    return X_base_tr, y_base_tr, K_inv_decayed, kernel, kernel_func, kernel_args, noise, epoch_list, r2_list_tr, mse_list_tr, r2_list_vl, mse_list_vl




def dao_gp2(X_train, y_train, INITIAL_BATCH_SIZE, INCREMENT_SIZE, DECAY_GAMMA, MAX_INDUCING, INITIAL_KERNEL, KPI, Z, SAFE_AREA_THRESHOLD,
           KERNEL_POOL, UNCERTAINTY_THRESHOLD, STRETCH_FACTOR=None):
    n_samples, n_features = X_train.shape
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()

    mbPerformanceManagerTr = MBPerformanceManager()
    mbPerformanceManagerVl = MBPerformanceManager()
    time_counter = 0


    X_base = X_train[:INITIAL_BATCH_SIZE]
    y_base = y_train[:INITIAL_BATCH_SIZE]

    VALIDATION_FRACTION = .5
    if len(X_base) == 1:
        X_base_tr = X_base_vl = X_base
        y_base_tr = y_base_vl = y_base
    else:
        X_base_tr, X_base_vl, y_base_tr, y_base_vl = train_test_split(X_base, y_base, test_size=VALIDATION_FRACTION)
    # X_base_tr = X_base_vl = X_base
    # y_base_tr = y_base_vl = y_base
    X_base_time = Util.append_timestamps([], len(X_base_tr), time_counter)
    # print(X_base_time)
    kernel, kernel_params, opt_dict, kernel_func, best_kernel_args, best_r2, best_mse, best_r2_tr, best_mse_tr = \
        KernelPoolHandler.pick_best_kernel(INITIAL_KERNEL, X_base_tr, y_base_tr, X_base_vl, y_base_vl, KERNEL_POOL, n_features, STRETCH_FACTOR, KPI, tol=1e-6)



    noise = opt_dict["noise"]
    kernel_args = {k: v for k, v in opt_dict.items() if k != "noise"}
    K = kernel_func(X_base_tr, X_base_tr, **kernel_args) + noise * np.eye(len(y_base_tr))
    K_inv_undecayed = np.linalg.inv(K)

    # Statistical Meta-Data Saved about First Mini-Batch (BaseModel)
    miniBatchMetaData = MiniBatchMetaData(best_r2, best_mse)
    memoryManager.add_mini_batch_data(miniBatchMetaData)

    # PerformanceRecorded for Trn and Vl
    mbPerformanceVl = MBPerformance(INITIAL_BATCH_SIZE,best_r2, best_mse)
    mbPerformanceManagerVl.add_mini_batch_data(mbPerformanceVl)
    mbPerformanceTr = MBPerformance(INITIAL_BATCH_SIZE,best_r2_tr, best_mse_tr)
    mbPerformanceManagerTr.add_mini_batch_data(mbPerformanceTr)


    for i in range(INITIAL_BATCH_SIZE, n_samples, INCREMENT_SIZE):
        time_counter+=1
        if time_counter > 1:
            K_inv_undecayed = K_inv_decayed
        print("**********************************ITERATION ", i, "***********************************")

        X_inc = X_train[i: i + INCREMENT_SIZE]
        y_inc = y_train[i: i + INCREMENT_SIZE]

        if len(X_inc) == 1:
            X_inc_tr = X_inc_vl = X_inc
            y_inc_tr = y_inc_vl = y_inc
        else:
            X_inc_tr, X_inc_vl, y_inc_tr, y_inc_vl = train_test_split(X_inc, y_inc, test_size=VALIDATION_FRACTION)
        # X_inc_tr = X_inc_vl = X_inc
        # y_inc_tr = y_inc_vl = y_inc
        for x_new, y_new in zip(X_inc_tr, y_inc_tr):
            predictive_var = Util.compute_predictive_variance(X_window=X_base_tr, y_window=y_base_tr, X_base_time=X_base_time, gamma=DECAY_GAMMA,
                                                              K_inv_undecayed=K_inv_undecayed,
                                                              x_new=x_new, noise=noise,
                                                              kernel_func=kernel_func, **kernel_args)
            if predictive_var > UNCERTAINTY_THRESHOLD:
                x_new = np.atleast_2d(x_new)
                # Attempt Woodbury update.
                K_inv_updated = Util.woodbury_update(K_inv_undecayed, X_base_tr, x_new, kernel_func, kernel_args, noise)
                if K_inv_updated is not None: # is woodbury stable
                    K_inv_undecayed = K_inv_updated
                    X_base_tr = np.vstack([X_base_tr, x_new])
                    y_base_tr = np.append(y_base_tr, y_new)
                    X_base_time = Util.append_timestamps(X_base_time, len(x_new), time_counter)
                else:
                    X_base_tr = np.vstack([X_base_tr, x_new])
                    y_base_tr = np.append(y_base_tr, y_new)
                    X_base_time = Util.append_timestamps(X_base_time, len(x_new), time_counter)
                    K = kernel_func(X_base_tr, X_base_tr, **kernel_args) + noise  * np.eye(len(y_base_tr))
                    K_inv_undecayed = np.linalg.inv(K)



        min_window_length = 7
        # if len(memoryManager.mini_batch_data) < min_window_length:
        #     K, K_inv_undecayed = Util.optimizeKernelHyperparamsGeneric(X_base_tr, y_base_tr, kernel, kernel_params, kernel_func)

        # Predict and Evaluate on 1. Train 2. Validation #####################
        mu_train, _ = Util.computeGP(X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv_undecayed,
                                         X_star=X_base_tr, noise=noise,
                                         kernel_func=kernel_func, **kernel_args)

        # mu_train, _ = Util.compute_weighted_GP(X_train=X_base_tr, y_train=y_base_tr, X_train_time=X_base_time, gamma=DECAY_GAMMA, X_star=X_base_tr, noise=noise, kernel_func=kernel_func, **kernel_args)
        # put new one with optimization
        mse_train = mean_squared_error(y_base_tr, mu_train)
        r2_train = r2_score(y_base_tr, mu_train)
        print(f"Update {i}: TRAIN MSE={mse_train:.4f}, R^2={r2_train:.4f}")

        # Also evaluate on the Validation Set.
        val_pred, _ = Util.computeGP(X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv_undecayed,
                                         X_star=X_inc_vl, noise=noise,
                                         kernel_func=kernel_func, **kernel_args)
        # val_pred, _ = Util.compute_weighted_GP(X_train=X_base_tr, y_train=y_base_tr, X_train_time=X_base_time, gamma=DECAY_GAMMA, X_star=X_inc_vl, noise=noise, kernel_func=kernel_func, **kernel_args)
        val_mse = mean_squared_error(y_inc_vl, val_pred)
        val_r2 = r2_score(y_inc_vl, val_pred)
        print("RES: Validation : ", "MSE", val_mse, "R^2", val_r2)
        ############# End Prediction and Evaluation ############################

        miniBatchMetaData = MiniBatchMetaData(val_r2, val_mse)
        memoryManager.add_mini_batch_data(miniBatchMetaData)

        # Performance Recorded
        mbPerformanceTr = MBPerformance(i + INCREMENT_SIZE,r2_train, mse_train)
        mbPerformanceManagerTr.add_mini_batch_data(mbPerformanceTr)
        mbPerformanceVl = MBPerformance(i + INCREMENT_SIZE,val_r2, val_mse)
        mbPerformanceManagerVl.add_mini_batch_data(mbPerformanceVl)

        # Concept Drift
        min_window_length = 7  # duplicated intentionally here.
        max_window_length = 31
        is_drift, drift_type = ConceptDriftHandler.evaluate_cocept_drift(conceptDriftDetector, memoryManager, KPI, Z,
                                                                         min_window_length, max_window_length,
                                                                         SAFE_AREA_THRESHOLD)
        print("is_drift", is_drift, "drift_type", drift_type)
        if is_drift:
            print('##### is_drift ', is_drift, "#####")
            optimized_dict = KernelHyperparameterOptimizer.optimize_hyperparameters(X_base_tr, y_base_tr, kernel,
                                                                                    kernel_params)
            noise = optimized_dict["noise"]
            kernel_args = {k: v for k, v in optimized_dict.items() if k != "noise"}
            K = kernel_func(X_base_tr, X_base_tr, **kernel_args) + noise * np.eye(len(y_base_tr))
            K_inv_undecayed = np.linalg.inv(K)  # Check if we can apply woodbury here to avoid the inverse?

            mu_train, _ = Util.computeGP(X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv_undecayed,
                                             X_star=X_base_tr, noise=noise,
                                             kernel_func=kernel_func, **kernel_args)
            # mu_train, _ = Util.compute_weighted_GP(X_train=X_base_tr, y_train=y_base_tr, X_train_time=X_base_time, gamma=DECAY_GAMMA, X_star=X_base_tr, noise=noise, kernel_func=kernel_func, **kernel_args)
            # put new one with optimization
            mse_train = mean_squared_error(y_base_tr, mu_train)
            r2_train = r2_score(y_base_tr, mu_train)
            print(f"Reopt at update {i}: TRAIN MSE={mse_train:.4f}, R^2={r2_train:.4f}")

            # Evaluate on validation set.
            val_pred, _ = Util.computeGP(X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv_undecayed,
                                             X_star=X_inc_vl, noise=noise,
                                             kernel_func=kernel_func, **kernel_args)
            # val_pred, _ = Util.compute_weighted_GP(X_train=X_base_tr, y_train=y_base_tr, X_train_time=X_base_time, gamma=DECAY_GAMMA, X_star=X_inc_vl, noise=noise, kernel_func=kernel_func, **kernel_args)
            val_mse = mean_squared_error(y_inc_vl, val_pred)
            val_r2 = r2_score(y_inc_vl, val_pred)
            print(f"Reopt at update {i}: Validation MSE={val_mse:.4f}, R^2={val_r2:.4f}")

            # remove latest entry
            memoryManager.remove_last_mini_batch_data()
            miniBatchMetaData = MiniBatchMetaData(val_r2, val_mse)
            memoryManager.add_mini_batch_data(miniBatchMetaData)

            # Performance Recorded
            mbPerformanceManagerTr.remove_last_mini_batch_data()
            mbPerformanceManagerVl.remove_last_mini_batch_data()
            mbPerformanceTr = MBPerformance(i + INCREMENT_SIZE,r2_train, mse_train)
            mbPerformanceManagerTr.add_mini_batch_data(mbPerformanceTr)
            mbPerformanceVl = MBPerformance(i + INCREMENT_SIZE, val_r2, val_mse)
            mbPerformanceManagerVl.add_mini_batch_data(mbPerformanceVl)

            # Reassess (update is_drift and drift_type)
            is_drift, drift_type = ConceptDriftHandler.evaluate_cocept_drift(conceptDriftDetector, memoryManager, KPI,
                                                                             Z, min_window_length, max_window_length)
            if is_drift and drift_type == "LT":
                print("*******LT*****")
                kernel, kernel_params, opt_dict, kernel_func, kernel_args, best_r2, best_mse, best_r2_tr, best_mse_tr = \
                    KernelPoolHandler.pick_best_kernel(kernel, X_base_tr, y_base_tr, X_inc_vl, y_inc_vl,
                                                       KERNEL_POOL, n_features, STRETCH_FACTOR, KPI)
                memoryManager.remove_last_mini_batch_data()
                miniBatchMetaData = MiniBatchMetaData(best_r2, best_mse)
                memoryManager.add_mini_batch_data(miniBatchMetaData)

                # PerformanceRecorded for Trn and Vl
                mbPerformanceManagerVl.remove_last_mini_batch_data()
                mbPerformanceManagerTr.remove_last_mini_batch_data()
                mbPerformanceVl = MBPerformance(i + INCREMENT_SIZE, best_r2, best_mse)
                mbPerformanceManagerVl.add_mini_batch_data(mbPerformanceVl)
                mbPerformanceTr = MBPerformance(i + INCREMENT_SIZE, best_r2_tr, best_mse_tr)
                mbPerformanceManagerTr.add_mini_batch_data(mbPerformanceTr)


        # print("before select inducing points X_base_tr", X_base_tr, "y_base_tr", y_base_tr, "X_base_time", X_base_time)

        X_base_tr, y_base_tr, K_inv_decayed, X_base_time = InducingPoints.select_inducing_points(
                                                    X_base_tr, y_base_tr, X_base_tr, noise,
                                                    DECAY_GAMMA,kernel_func, kernel_args, MAX_INDUCING, X_base_time)

        # print("after select inducing points X_base_tr", X_base_tr, "y_base_tr", y_base_tr, "X_base_time", X_base_time)

        # K_inv_decayed = Util.apply_decay_with_time(K_inv_undecayed, DECAY_GAMMA, X_base_time)

        mu_tr, _ = Util.computeGP(X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv_decayed,
                                           X_star=X_inc_tr, noise=noise,
                                           kernel_func=kernel_func, **kernel_args)
        # mu_tr, _ = Util.compute_weighted_GP(X_train= X_base_tr, y_train=y_base_tr, X_train_time=X_base_time, gamma=DECAY_GAMMA, X_star=X_inc_tr, noise=noise, kernel_func=kernel_func, ** kernel_args)

        mse_tr = mean_squared_error(y_inc_tr, mu_tr)
        r2_tr = r2_score(y_inc_tr, mu_tr)
        print(f"Update (after selecting inducing points) {i + 1} | TRAINING MSE: {mse_tr:.4f}, R^2: {r2_tr:.4f}")

        mu_vl, _ = Util.computeGP(X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv_decayed,
                                         X_star=X_inc_vl, noise=noise,
                                         kernel_func=kernel_func, **kernel_args)
        # mu_vl, _ = Util.compute_weighted_GP(X_train= X_base_tr, y_train=y_base_tr, X_train_time=X_base_time, gamma=DECAY_GAMMA, X_star=X_inc_vl, noise=noise, kernel_func=kernel_func, ** kernel_args)

        mse_val = mean_squared_error(y_inc_vl, mu_vl)
        r2_val = r2_score(y_inc_vl, mu_vl)
        print(f"Update (after selecting inducing points) {i + 1} | VALIDATION MSE: {mse_val:.4f}, R^2: {r2_val:.4f}")

    # PerformanceRecorder
    epoch_list = mbPerformanceManagerTr.get_epochs_list()
    r2_list_tr = mbPerformanceManagerTr.get_r2_list()
    mse_list_tr = mbPerformanceManagerTr.get_mse_list()
    r2_list_vl = mbPerformanceManagerVl.get_r2_list()
    mse_list_vl = mbPerformanceManagerVl.get_mse_list()
    print("Epochs")
    Printer.print_list_tabulate(epoch_list)
    print("Training")
    Printer.print_list_tabulate(r2_list_tr)
    print("Validation")
    Printer.print_list_tabulate(r2_list_vl)

    return X_base_tr, y_base_tr, K_inv_decayed, kernel, kernel_func, kernel_args, noise, epoch_list, r2_list_tr, mse_list_tr, r2_list_vl, mse_list_vl

