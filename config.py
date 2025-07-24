def get_config():
    hyperparameters = [
        {
            "model_name": "m256k5v1",
            "dconv_input": 16,
            "dconv_output": 16,
            "pconv_input": 16,
            "pconv_hidden": 16,
            "pconv_output": 256,
            "kernel_size": 5,
            "epochs": 50
        },
        {
            "model_name": "m256k5v2",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 64,
            "pconv_output": 256,
            "kernel_size": 5,
            "epochs": 50
        },
        {
            "model_name": "m256k5v3",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 32,
            "pconv_output": 256,
            "kernel_size": 5,
            "epochs": 50
        },
        {
            "model_name": "m256k10v1",
            "dconv_input": 16,
            "dconv_output": 16,
            "pconv_input": 16,
            "pconv_hidden": 16,
            "pconv_output": 256,
            "kernel_size": 10,
            "epochs": 50
        },
        {
            "model_name": "m256k10v2",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 64,
            "pconv_output": 256,
            "kernel_size": 10,
            "epochs": 50
        },
        {
            "model_name": "m256k10v3",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 32,
            "pconv_output": 256,
            "kernel_size": 10,
            "epochs": 50
        },
                {
            "model_name": "m256k15v1",
            "dconv_input": 16,
            "dconv_output": 16,
            "pconv_input": 16,
            "pconv_hidden": 16,
            "pconv_output": 256,
            "kernel_size": 15,
            "epochs": 50
        },
        {
            "model_name": "m256k15v2",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 64,
            "pconv_output": 256,
            "kernel_size": 15,
            "epochs": 50
        },
        {
            "model_name": "m256k15v3",
            "dconv_input": 32,
            "dconv_output": 32,
            "pconv_input": 32,
            "pconv_hidden": 32,
            "pconv_output": 256,
            "kernel_size": 15,
            "epochs": 50
        }
    ]

    return hyperparameters
