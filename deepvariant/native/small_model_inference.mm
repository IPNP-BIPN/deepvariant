// Obj-C++ wrapper around the small_model .mlpackage.
#include "deepvariant/native/small_model_inference.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cstring>
#include <memory>
#include <string>

namespace deepvariant {

struct SmallModel::Impl {
  MLModel* model = nil;
  NSString* input_name = @"input_1";
  NSString* output_name = @"Identity";
};

SmallModel::SmallModel() : impl_(std::make_unique<Impl>()) {}
SmallModel::~SmallModel() = default;

// static
std::unique_ptr<SmallModel> SmallModel::Load(const std::string& path) {
  @autoreleasepool {
    NSError* error = nil;
    NSURL* url = [NSURL fileURLWithPath:
        [NSString stringWithUTF8String:path.c_str()]];
    NSURL* compiled = [MLModel compileModelAtURL:url error:&error];
    if (!compiled) {
      NSLog(@"SmallModel compile failed: %@", error.localizedDescription);
      return nullptr;
    }
    MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = MLComputeUnitsAll;
    MLModel* model = [MLModel modelWithContentsOfURL:compiled
                                        configuration:cfg
                                                error:&error];
    if (!model) {
      NSLog(@"SmallModel load failed: %@", error.localizedDescription);
      return nullptr;
    }

    auto out = std::unique_ptr<SmallModel>(new SmallModel());
    out->impl_->model = model;
    MLModelDescription* desc = model.modelDescription;
    if (desc.inputDescriptionsByName.count > 0) {
      out->impl_->input_name = desc.inputDescriptionsByName.allKeys.firstObject;
    }
    if (desc.outputDescriptionsByName.count > 0) {
      out->impl_->output_name = desc.outputDescriptionsByName.allKeys.firstObject;
    }
    return out;
  }
}

bool SmallModel::Predict(const float* features, int N, float* probs) {
  @autoreleasepool {
    NSError* error = nil;
    MLModel* model = impl_->model;
    NSString* in_name  = impl_->input_name;
    NSString* out_name = impl_->output_name;

    for (int i = 0; i < N; ++i) {
      // Shape (1, 70) — model expects 2-D input.
      NSArray<NSNumber*>* shape = @[@1, @70];
      MLMultiArray* arr = [[MLMultiArray alloc]
          initWithShape:shape
              dataType:MLMultiArrayDataTypeFloat32
                 error:&error];
      if (!arr) return false;
      std::memcpy(arr.dataPointer, features + i * 70, 70 * sizeof(float));

      MLDictionaryFeatureProvider* fp =
          [[MLDictionaryFeatureProvider alloc]
              initWithDictionary:@{in_name: arr}
                           error:&error];
      if (!fp) return false;

      id<MLFeatureProvider> result =
          [model predictionFromFeatures:fp error:&error];
      if (!result) return false;

      MLMultiArray* out_arr =
          [result featureValueForName:out_name].multiArrayValue;
      if (!out_arr) return false;
      std::memcpy(probs + i * 3, out_arr.dataPointer, 3 * sizeof(float));
    }
    return true;
  }
}

}  // namespace deepvariant
