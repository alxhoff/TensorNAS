def mlonmcu_evaluate_model(
    test_name=None,
    model_name=None,
    logger=None,
    metrics=["Cycles", "Total ROM", "Total RAM"],
    platform=["mlif"],
    backend=["tvmaotplus"],
    target=["etiss_pulpino"],
    frontend=["tflite"],
    postprcess=None,
    feature=None,
    configs=None,
    parllel=None,
    progress=False,
    verbose=False,
):
    if logger:
        logger.log("MLonMCU evaluating model, name:{}".format(model_name))

    import os

    path = "Output/{}/Models/{}/".format(test_name, model_name)

    if not os.path.isdir(path):
        print("INVALID MODEL PATH: the following path not found: ", path)

    run_mlonmcu_flow(
        path=path,
        platform=platform,
        backend=backend,
        target=target,
        frontend=frontend,
        postprcess=postprcess,
        feature=feature,
        configs=configs,
        parllel=parllel,
        progress=progress,
        verbose=verbose,
    )

    import csv

    # read the run_metrics.csv
    run_metrics_lines = []
    with open(path + "/mlonmcu_out/run_metrics.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            run_metrics_lines.append(line)

    # organize the run_metrics.csv in a dictionary
    mlonmcu_metrics = {
        metric: val for metric, val in zip(run_metrics_lines[0], run_metrics_lines[1])
    }

    # prepare a list of return values
    ret = []
    for val in metrics:
        if val in mlonmcu_metrics:
            ret.append(float(mlonmcu_metrics[val]))
        else:
            raise Exception("required metric not supported by MLonMCU: ", val)

    return ret


def run_mlonmcu_flow(
    path=None,
    platform=None,
    backend=None,
    target=None,
    frontend=None,
    postprcess=None,
    feature=None,
    configs=None,  # KEY=VALUE
    parllel=False,  # Use multiple threads to process runs in parallel (8 if specified, else 1)
    progress=False,  # Display progress bar
    verbose=False,
):
    import os
    import subprocess

    run_flow_args = []

    if not path:
        raise Exception("No model path specified")

    import subprocess as p

    if platform:
        for i in range(len(platform)):
            run_flow_args.append("--platform")
            run_flow_args.append(platform[i])

    if backend:
        for i in range(len(backend)):
            run_flow_args.append("--backend")
            run_flow_args.append(backend[i])

    if target:
        for i in range(len(target)):
            run_flow_args.append("--target")
            run_flow_args.append(target[i])

    if frontend:
        for i in range(len(frontend)):
            run_flow_args.append("--frontend")
            run_flow_args.append(frontend[i])

    if postprcess:
        for i in range(len(postprcess)):
            run_flow_args.append("--postprocess")
            run_flow_args.append(postprcess[i])

    if feature:
        for i in range(len(feature)):
            run_flow_args.append("--feature")
            run_flow_args.append(feature[i])

    if configs:
        for i in range(len(configs)):
            run_flow_args.append("--config")
            run_flow_args.append(configs[i])

    if parllel == True:
        run_flow_args.append("--parallel")
        run_flow_args.append(str(parllel))

    if progress == True:
        run_flow_args.append("--progress")

    if verbose == True:
        run_flow_args.append("--verbose")

    path_prefix = "Demos/DEAP/"
    run_flow_args = " ".join(run_flow_args)
    project_dir = "{}/../../".format(os.getcwd())
    export_dir = "/mount/{}{}mlonmcu_out/".format(path_prefix, path)
    local_dir = "{}{}mlonmcu_out".format(path_prefix, path)
    os.makedirs(local_dir)

    flow_command = "mlonmcu flow run /mount/{}{}saved_model.tflite {}".format(
        path_prefix, path, run_flow_args
    )
    export_command = "mlonmcu export {} --run --force".format(export_dir)
    chmod_command = "chmod -R 777 {}".format(export_dir)

    docker_command = "docker run --entrypoint /bin/bash -v {}:/mount tumeda/mlonmcu-bench:latest -c '{} && {} && {}'".format(
        project_dir, flow_command, export_command, chmod_command
    )

    subprocess.call(docker_command, shell=True)
