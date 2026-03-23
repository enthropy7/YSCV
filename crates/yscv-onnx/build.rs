use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let generated = out_dir.join("onnx.rs");

    // Try to compile protos with protoc
    match prost_build::Config::new().compile_protos(&["proto/onnx-ml.proto"], &["proto/"]) {
        Ok(()) => {
            // Success — prost wrote onnx.rs to OUT_DIR
        }
        Err(e) => {
            eprintln!("cargo:warning=protoc not found ({e}), generating fallback stub");
            // Write a minimal fallback so the build doesn't fail.
            // The fallback contains empty structs for the types we use.
            // For full functionality, install protoc: https://grpc.io/docs/protoc-installation/
            let fallback = r#"
// Auto-generated fallback — install protoc for full ONNX proto support.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelProto {
    #[prost(int64, optional, tag = "1")]
    pub ir_version: ::core::option::Option<i64>,
    #[prost(string, optional, tag = "2")]
    pub producer_name: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(string, optional, tag = "3")]
    pub producer_version: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(string, optional, tag = "4")]
    pub domain: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(int64, optional, tag = "5")]
    pub model_version: ::core::option::Option<i64>,
    #[prost(string, optional, tag = "6")]
    pub doc_string: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(message, optional, tag = "7")]
    pub graph: ::core::option::Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: ::prost::alloc::vec::Vec<OperatorSetIdProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: ::prost::alloc::vec::Vec<NodeProto>,
    #[prost(string, optional, tag = "2")]
    pub name: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(message, repeated, tag = "5")]
    pub initializer: ::prost::alloc::vec::Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub input: ::prost::alloc::vec::Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: ::prost::alloc::vec::Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, repeated, tag = "2")]
    pub output: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, optional, tag = "3")]
    pub name: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(string, optional, tag = "4")]
    pub op_type: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(string, optional, tag = "7")]
    pub domain: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(message, repeated, tag = "5")]
    pub attribute: ::prost::alloc::vec::Vec<AttributeProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: ::prost::alloc::vec::Vec<i64>,
    #[prost(int32, optional, tag = "2")]
    pub data_type: ::core::option::Option<i32>,
    #[prost(string, optional, tag = "8")]
    pub name: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(bytes = "vec", optional, tag = "4")]
    pub raw_data: ::core::option::Option<::prost::alloc::vec::Vec<u8>>,
    #[prost(float, repeated, tag = "5")]
    pub float_data: ::prost::alloc::vec::Vec<f32>,
    #[prost(int32, repeated, tag = "6")]
    pub int32_data: ::prost::alloc::vec::Vec<i32>,
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: ::prost::alloc::vec::Vec<i64>,
    #[prost(double, repeated, tag = "10")]
    pub double_data: ::prost::alloc::vec::Vec<f64>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AttributeProto {
    #[prost(string, optional, tag = "1")]
    pub name: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(int32, optional, tag = "20")]
    pub r#type: ::core::option::Option<i32>,
    #[prost(float, optional, tag = "4")]
    pub f: ::core::option::Option<f32>,
    #[prost(int64, optional, tag = "3")]
    pub i: ::core::option::Option<i64>,
    #[prost(string, optional, tag = "5")]
    pub s: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(message, optional, tag = "6")]
    pub t: ::core::option::Option<TensorProto>,
    #[prost(float, repeated, tag = "7")]
    pub floats: ::prost::alloc::vec::Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: ::prost::alloc::vec::Vec<i64>,
    #[prost(string, repeated, tag = "9")]
    pub strings: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OperatorSetIdProto {
    #[prost(string, optional, tag = "1")]
    pub domain: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(int64, optional, tag = "2")]
    pub version: ::core::option::Option<i64>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ValueInfoProto {
    #[prost(string, optional, tag = "1")]
    pub name: ::core::option::Option<::prost::alloc::string::String>,
    #[prost(message, optional, tag = "2")]
    pub r#type: ::core::option::Option<TypeProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: ::core::option::Option<type_proto::Tensor>,
}

pub mod type_proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Tensor {
        #[prost(int32, optional, tag = "1")]
        pub elem_type: ::core::option::Option<i32>,
        #[prost(message, optional, tag = "2")]
        pub shape: ::core::option::Option<super::TensorShapeProto>,
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: ::prost::alloc::vec::Vec<tensor_shape_proto::Dimension>,
}

pub mod tensor_shape_proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Dimension {
        #[prost(string, optional, tag = "3")]
        pub denotation: ::core::option::Option<::prost::alloc::string::String>,
        #[prost(oneof = "dimension::Value", tags = "1, 2")]
        pub value: ::core::option::Option<dimension::Value>,
    }
    pub mod dimension {
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum Value {
            #[prost(int64, tag = "1")]
            DimValue(i64),
            #[prost(string, tag = "2")]
            DimParam(::prost::alloc::string::String),
        }
    }
}
"#;
            std::fs::write(&generated, fallback).expect("failed to write fallback onnx.rs");
        }
    }
}
