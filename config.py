def get_config():
    hyperparameters = [
        {
            "model_name": "model_256_256_5",
            "dconv_input": 1,
            "dconv_output": 1,
            "pconv_input": 1,
            "pconv_hidden": 256,
            "pconv_output": 256,
            "kernel_size": 5,
            "epochs": 50
        },
        {
            "model_name": "model_256_256_10",
            "dconv_input": 1,
            "dconv_output": 1,
            "pconv_input": 1,
            "pconv_hidden": 256,
            "pconv_output": 256,
            "kernel_size": 10,
            "epochs": 50
        },
        {
            "model_name": "model_256_256_15",
            "dconv_input": 1,
            "dconv_output": 1,
            "pconv_input": 1,
            "pconv_hidden": 256,
            "pconv_output": 256,
            "kernel_size": 15,
            "epochs": 50
        },
        {
            "model_name": "model_256_512_5",
            "dconv_input": 1,
            "dconv_output": 1,
            "pconv_input": 1,
            "pconv_hidden": 256,
            "pconv_output": 512,
            "kernel_size": 5,
            "epochs": 50
        },
        {
            "model_name": "model_256_512_10",
            "dconv_input": 1,
            "dconv_output": 1,
            "pconv_input": 1,
            "pconv_hidden": 256,
            "pconv_output": 512,
            "kernel_size": 10,
            "epochs": 50
        },
        {
            "model_name": "model_256_512_15",
            "dconv_input": 1,
            "dconv_output": 1,
            "pconv_input": 1,
            "pconv_hidden": 256,
            "pconv_output": 512,
            "kernel_size": 15,
            "epochs": 50
        },
        {
            "model_name": "model_512_256_5",
            "dconv_input": 1,
            "dconv_output": 1,
            "pconv_input": 1,
            "pconv_hidden": 512,
            "pconv_output": 256,
            "kernel_size": 5,
            "epochs": 50
        },
        {
            "model_name": "model_512_256_10",
            "dconv_input": 1,
            "dconv_output": 1,
            "pconv_input": 1,
            "pconv_hidden": 512,
            "pconv_output": 256,
            "kernel_size": 10,
            "epochs": 50
        },
        {
            "model_name": "model_512_256_15",
            "dconv_input": 1,
            "dconv_output": 1,
            "pconv_input": 1,
            "pconv_hidden": 512,
            "pconv_output": 256,
            "kernel_size": 15,
            "epochs": 50
        },
    ]

    return hyperparameters
