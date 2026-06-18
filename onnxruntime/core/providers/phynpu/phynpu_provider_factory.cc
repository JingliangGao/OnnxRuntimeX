 
//Data: 2026-01-08 

#include "core/providers/phynpu/phynpu_provider_factory.h"
#include "phynpu_execution_provider.h"
#include "phynpu_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct PHYnpuProviderFactory : IExecutionProviderFactory {
  PHYnpuProviderFactory() {}
  ~PHYnpuProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> PHYnpuProviderFactory::CreateProvider() {
  return std::make_unique<PHYnpuExecutionProvider>();
}

std::shared_ptr<IExecutionProviderFactory>
PHYnpuProviderFactoryCreator::Create() {
  return std::make_shared<onnxruntime::PHYnpuProviderFactory>();
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_PHYNPU,
                    _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(
      onnxruntime::PHYnpuProviderFactoryCreator::Create());
  return nullptr;
}
