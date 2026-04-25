// Core ML inference implementation (Obj-C++).
// Loads a .mlpackage, compiles on first run (Core ML caches the
// .mlmodelc in ~/Library/Caches/com.apple.CoreML/), and runs
// batched prediction via MLModel.predictionsFromBatch:error:.

#include "deepvariant/native/coreml_inference.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace deepvariant {

struct CoreMLModel::Impl {
  MLModel* model = nil;
  NSString* input_name  = @"x";
  NSString* output_name = @"classification";
};

CoreMLModel::CoreMLModel() : impl_(std::make_unique<Impl>()) {}
CoreMLModel::~CoreMLModel() = default;

// static
std::unique_ptr<CoreMLModel> CoreMLModel::Load(
    const std::string& path, ComputeUnits compute_units) {
  @autoreleasepool {
    NSError* error = nil;
    NSURL* url = [NSURL fileURLWithPath:
        [NSString stringWithUTF8String:path.c_str()]];

    // Compile the .mlpackage to .mlmodelc (cached by Core ML).
    NSURL* compiled = [MLModel compileModelAtURL:url error:&error];
    if (!compiled) {
      NSLog(@"CoreML compile failed: %@", error.localizedDescription);
      return nullptr;
    }

    MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
    switch (compute_units) {
      case ComputeUnits::kAll:
        cfg.computeUnits = MLComputeUnitsAll;
        break;
      case ComputeUnits::kCpuAndGpu:
        cfg.computeUnits = MLComputeUnitsCPUAndGPU;
        break;
      case ComputeUnits::kCpuOnly:
        cfg.computeUnits = MLComputeUnitsCPUOnly;
        break;
    }

    MLModel* model = [MLModel modelWithContentsOfURL:compiled
                                        configuration:cfg
                                                error:&error];
    if (!model) {
      NSLog(@"CoreML load failed: %@", error.localizedDescription);
      return nullptr;
    }

    // Inspect input/output names + shapes from the model description.
    auto out = std::unique_ptr<CoreMLModel>(new CoreMLModel());
    out->impl_->model = model;

    MLModelDescription* desc = model.modelDescription;
    if (desc.inputDescriptionsByName.count > 0) {
      NSString* name = desc.inputDescriptionsByName.allKeys.firstObject;
      out->impl_->input_name = name;
      out->input_name_ = name.UTF8String;
      MLFeatureDescription* fd = desc.inputDescriptionsByName[name];
      if (fd.type == MLFeatureTypeMultiArray) {
        NSArray<NSNumber*>* shape = fd.multiArrayConstraint.shape;
        if (shape.count >= 4) {
          // shape = (N, H, W, C) or (N, C, H, W); our model uses NHWC.
          out->input_height_   = shape[1].intValue;
          out->input_width_    = shape[2].intValue;
          out->input_channels_ = shape[3].intValue;
        }
      }
    }
    if (desc.outputDescriptionsByName.count > 0) {
      NSString* name = desc.outputDescriptionsByName.allKeys.firstObject;
      out->impl_->output_name = name;
      out->output_name_ = name.UTF8String;
      MLFeatureDescription* fd = desc.outputDescriptionsByName[name];
      if (fd.type == MLFeatureTypeMultiArray) {
        NSArray<NSNumber*>* shape = fd.multiArrayConstraint.shape;
        if (shape.count >= 2) {
          out->num_classes_ = shape[1].intValue;
        }
      }
    }

    return out;
  }
}

bool CoreMLModel::Predict(const float* images, int N, int H, int W, int C,
                          float* probs, int num_classes) {
  @autoreleasepool {
    NSError* error = nil;
    MLModel* model = impl_->model;
    NSString* in_name  = impl_->input_name;
    NSString* out_name = impl_->output_name;

    const NSInteger elemPerImage = H * W * C;

    // Build a batch of MLFeatureProvider, one per example.
    // Run inference example by example.
    // (Batched MLArrayBatchProvider API was deprecated in macOS 14.
    //  Single-prediction loop is compatible and correct.)
    for (int i = 0; i < N; ++i) {
      NSArray<NSNumber*>* shape = @[@(H), @(W), @(C)];
      MLMultiArray* arr = [[MLMultiArray alloc]
          initWithShape:shape
              dataType:MLMultiArrayDataTypeFloat32
                 error:&error];
      if (!arr) {
        NSLog(@"MLMultiArray alloc failed: %@", error.localizedDescription);
        return false;
      }
      float* dst = (float*)arr.dataPointer;
      std::memcpy(dst, images + (size_t)i * elemPerImage,
                  (size_t)elemPerImage * sizeof(float));

      MLDictionaryFeatureProvider* fp =
          [[MLDictionaryFeatureProvider alloc]
              initWithDictionary:@{in_name: arr}
                           error:&error];
      if (!fp) {
        NSLog(@"Feature provider failed: %@", error.localizedDescription);
        return false;
      }

      id<MLFeatureProvider> result =
          [model predictionFromFeatures:fp error:&error];
      if (!result) {
        NSLog(@"Prediction failed for example %d: %@", i,
              error.localizedDescription);
        return false;
      }

      MLMultiArray* out_arr =
          [result featureValueForName:out_name].multiArrayValue;
      if (!out_arr) {
        NSLog(@"Output '%@' missing in result %d", out_name, i);
        return false;
      }
      const float* src = (const float*)out_arr.dataPointer;
      std::memcpy(probs + (size_t)i * num_classes, src,
                  (size_t)num_classes * sizeof(float));
    }
    return true;
  }
}

}  // namespace deepvariant
