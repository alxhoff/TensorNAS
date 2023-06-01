def mlonmcu_evaluate_model(
    # self,
    model=None,
    test_name=None,
    model_name=None,
    logger=None,
    metrics=[
        "Cycles",
        "_MIPS_",
        "Total ROM",
        "Total RAM",
        "ROM read-only",
        "ROM code",
        "ROM misc",
        "RAM data",
        "RAM zero-init data",
        "_Run Stage Time [s]_",
    ],
    platform="mlif",
    backend="tvmaotplus",
    target="etiss_pulpino",
    frontend="tflite",
    verbose=False,
    # use_clear_memory=False,
):
    if logger:
        logger.log("MLonMCU evaluating model, name:{}".format(model_name))

    import os

    path = "Output/{}/Models/{}/".format(test_name, model_name)

    if not os.path.isdir(path):
        print("INVALID MODEL PATH: the following path not found: ", path)

    run_mlonmcu_flow(
        path=path, platform=platform, backend=backend, target=target, frontend=frontend
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

    print("___DEBUG________________________________________________________________")
    # print("model path: ", path)
    # print(run_metrics_lines[0])
    # print(run_metrics_lines[1])
    # print(mlonmcu_metrics)
    # print("ret: ",ret)
    print("________________________________________________________________________")

    return ret


"""
/__TO DO__/
- CREATRE A FUNCTION THAT QUANTIZE A MODEL BEFORE EVALUATING IT
- ADJUST THE FUNCTION TO RECEIVE MULTI BACKEND, TARGETS ...
- CONTROL INPUT ARGS VALIDITY
- PACK THE "MLONMCU FLOW RUN" IN SEPARATE FUNCTION
- MORE ORGANIZATON FOR THE MLONMCU EXPORT: MAKE SPECIFIC DIR FOR EACH TARGET WHICH CONTAINS DIRS FOR EACH BACKEND ...
- LOOK FOR MLONMCU OTHER FEATURES YOU CAN ADD TO THIS FUNCTION
- check supported platforms, backends and targets by mlonmcu
    mlonmcu_supported_platforms =[]
    mlonmcu_supported_backends =[]
    mlonmcu_supported_targets =[]
    ans = p.run(['mlonmcu', 'flow', '--list-targets'], capture_output=True)
    ans = ans.stdout.decode()
    --> check it only one time in configparse
-/ RETURN THE MTERICS VLAUES ASKED FOR
-/ WHEN THE "mlonmcu_out" DIR IS ALREADY THERE THE MLONMCU ASKS TO OVERWRITE IT --> MAKE IT AUTOMATED
"""


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
):
    run_flow_args = []

    if not path:
        raise Exception("No model path specified")

    import subprocess as p

    if platform:
        run_flow_args.append("--platform")
        run_flow_args.append(platform)

    if backend:
        run_flow_args.append("--backend")
        run_flow_args.append(backend)

    if target:
        run_flow_args.append("--target")
        run_flow_args.append(target)

    if frontend:
        run_flow_args.append("--frontend")
        run_flow_args.append(frontend)

    # run the mlonmcu flow to evaluate the model
    p.run(["mlonmcu", "flow", "run", path + "/saved_model.tflite", *run_flow_args])
    # export the mlonmcu output data to new directory "mlonmcu_out" in saved_model.tflite root
    p.run(["mlonmcu", "export", path + "/mlonmcu_out/", "--run", "--force"])

    print("___DEBUG________________________________________________________________")
    # print("run_flow_args:")
    # print(run_flow_args)
    print("________________________________________________________________________")
