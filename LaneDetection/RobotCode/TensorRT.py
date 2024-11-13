import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTModel:
    def __init__(self, onnx_model_path, trt_engine_path):
        self.onnx_model_path = onnx_model_path
        self.trt_engine_path = trt_engine_path
        self.engine = self.build_or_load_engine()
        self.context = self.engine.create_execution_context()
        
        #the horror I have to allocate memory 
        #this isn't C
        #what the heck
        self.input_shape = (1, 3, 64, 64)  
        self.output_shape = (1, 2)  
        self.d_input = cuda.mem_alloc(np.prod(self.input_shape) * np.float32().nbytes)
        self.d_output = cuda.mem_alloc(np.prod(self.output_shape) * np.float32().nbytes)

    def build_or_load_engine(self):
        try:
            with open(self.trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                print("Using existing TensorRT engine.")
                return runtime.deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            print("Building TensorRT engine from ONNX model.")
            return self.build_engine()

    def build_engine(self):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 
            with open(self.onnx_model_path, 'rb') as model:
                if not parser.parse(model.read()):
                    raise Exception("Failed to parse the ONNX file")

            engine = builder.build_cuda_engine(network)
            with open(self.trt_engine_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    def predict(self, input_data):
        cuda.memcpy_htod(self.d_input, input_data)
        self.context.execute_v2(bindings=[int(self.d_input), int(self.d_output)])
        h_output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(h_output, self.d_output)
        return h_output
