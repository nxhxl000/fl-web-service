export type Optimizer = 'sgd' | 'adamw'

export type ModelHParams = {
  optimizer: Optimizer
  'client-lr': number
  'client-momentum': number
  'client-weight-decay': number
  'batch-size': number
}

export type FlModelDef = {
  id: string
  label: string
  description: string
  defaults: ModelHParams
}

export const FL_MODELS: FlModelDef[] = [
  {
    id: 'wrn_16_4',
    label: 'WideResNet 16-4',
    description: 'CIFAR-100',
    defaults: {
      optimizer: 'sgd',
      'client-lr': 0.03,
      'client-momentum': 0.9,
      'client-weight-decay': 5e-4,
      'batch-size': 64,
    },
  },
  {
    id: 'se_resnet',
    label: 'SE-ResNet (custom)',
    description: 'CIFAR-100',
    defaults: {
      optimizer: 'sgd',
      'client-lr': 0.03,
      'client-momentum': 0.9,
      'client-weight-decay': 5e-4,
      'batch-size': 64,
    },
  },
  {
    id: 'effnet_b0',
    label: 'EfficientNet-B0',
    description: 'PlantVillage · trained from scratch',
    defaults: {
      optimizer: 'adamw',
      'client-lr': 1e-3,
      'client-momentum': 0,
      'client-weight-decay': 1e-2,
      'batch-size': 32,
    },
  },
]

export type ModelHParamNumericKey = Exclude<keyof ModelHParams, 'optimizer'>

export const MODEL_HPARAM_RANGES: Record<
  ModelHParamNumericKey,
  { min: number; max: number; step: number }
> = {
  'client-lr': { min: 0, max: 0.5, step: 0.001 },
  'client-momentum': { min: 0, max: 0.99, step: 0.01 },
  'client-weight-decay': { min: 0, max: 0.1, step: 0.0001 },
  'batch-size': { min: 1, max: 512, step: 1 },
}

export const MODEL_HPARAM_LABELS: Record<ModelHParamNumericKey, string> = {
  'client-lr': 'Learning rate',
  'client-momentum': 'Momentum (SGD only)',
  'client-weight-decay': 'Weight decay',
  'batch-size': 'Batch size',
}
