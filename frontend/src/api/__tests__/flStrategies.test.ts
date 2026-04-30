import { describe, expect, it } from 'vitest'
import {
  CIFAR_PER_STRATEGY,
  STRATEGY_GATED_KEYS,
  STRATEGY_SPECIFIC_KEYS,
  filterRunConfig,
  splitStrategyOverrides,
  type StrategyId,
} from '../flStrategies'

const ALL_STRATEGIES: StrategyId[] = ['fedavg', 'fedavgm', 'fedprox', 'fednovam']

describe('splitStrategyOverrides', () => {
  it('routes ModelHParams keys to hparamPatch', () => {
    const { hparamPatch, flParamPatch } = splitStrategyOverrides({
      'client-lr': 0.05,
      'client-momentum': 0.9,
    })
    expect(hparamPatch).toEqual({ 'client-lr': 0.05, 'client-momentum': 0.9 })
    expect(flParamPatch).toEqual({})
  })

  it('routes FLParams keys to flParamPatch', () => {
    const { hparamPatch, flParamPatch } = splitStrategyOverrides({
      'local-epochs': 3,
      'server-momentum': 0.5,
      'server-lr': 1.0,
      'proximal-mu': 0.0005,
    })
    expect(hparamPatch).toEqual({})
    expect(flParamPatch).toEqual({
      'local-epochs': 3,
      'server-momentum': 0.5,
      'server-lr': 1.0,
      'proximal-mu': 0.0005,
    })
  })

  it('handles mixed keys', () => {
    const { hparamPatch, flParamPatch } = splitStrategyOverrides(
      CIFAR_PER_STRATEGY.fedavgm,
    )
    expect(hparamPatch).toEqual({ 'client-lr': 0.05, 'client-momentum': 0.9 })
    expect(flParamPatch).toEqual({
      'local-epochs': 3,
      'server-momentum': 0.5,
      'server-lr': 1.0,
    })
  })

  it('returns empty patches for empty overrides', () => {
    expect(splitStrategyOverrides({})).toEqual({ hparamPatch: {}, flParamPatch: {} })
  })
})

describe('CIFAR_PER_STRATEGY', () => {
  it('mirrors fl_app/_CIFAR_PER_STRATEGY values for our 4 strategies', () => {
    expect(CIFAR_PER_STRATEGY.fedavg['client-lr']).toBe(0.1)
    expect(CIFAR_PER_STRATEGY.fedavgm['client-lr']).toBe(0.05)
    expect(CIFAR_PER_STRATEGY.fedprox['client-lr']).toBe(0.1)
    expect(CIFAR_PER_STRATEGY.fednovam['client-lr']).toBe(0.05)
  })

  it.each(ALL_STRATEGIES)('strategy %s has local-epochs = 3', (s) => {
    expect(CIFAR_PER_STRATEGY[s]['local-epochs']).toBe(3)
  })

  it.each(ALL_STRATEGIES)('strategy %s has client-momentum = 0.9', (s) => {
    expect(CIFAR_PER_STRATEGY[s]['client-momentum']).toBe(0.9)
  })

  it('only fedavgm and fednovam have server-momentum', () => {
    expect(CIFAR_PER_STRATEGY.fedavgm['server-momentum']).toBeDefined()
    expect(CIFAR_PER_STRATEGY.fednovam['server-momentum']).toBeDefined()
    expect(CIFAR_PER_STRATEGY.fedavg['server-momentum']).toBeUndefined()
    expect(CIFAR_PER_STRATEGY.fedprox['server-momentum']).toBeUndefined()
  })

  it('only fedprox has proximal-mu', () => {
    expect(CIFAR_PER_STRATEGY.fedprox['proximal-mu']).toBeDefined()
    expect(CIFAR_PER_STRATEGY.fedavg['proximal-mu']).toBeUndefined()
    expect(CIFAR_PER_STRATEGY.fedavgm['proximal-mu']).toBeUndefined()
    expect(CIFAR_PER_STRATEGY.fednovam['proximal-mu']).toBeUndefined()
  })
})

describe('filterRunConfig', () => {
  const FULL_CFG = {
    model: 'se_resnet',
    aggregation: 'fedavgm',
    'num-server-rounds': 80,
    'local-epochs': 3,
    'fraction-train': 1.0,
    'min-train-nodes': 10,
    'min-available-nodes': 10,
    'client-lr': 0.05,
    'client-momentum': 0.9,
    'client-weight-decay': 5e-4,
    'batch-size': 64,
    optimizer: 'sgd',
    'server-momentum': 0.5,
    'server-lr': 1.0,
    'proximal-mu': 0.0005,
    'straggler-mode': 'chunk',
    'straggler-target': 'min',
    'straggler-tolerance': 0.05,
    'straggler-drop-tolerance': 0.5,
    'straggler-max-dropped': 3,
    'straggler-min-chunk': 0.1,
    'straggler-min-epochs': 1,
  }

  it.each<[StrategyId, string[], string[]]>([
    ['fedavg',   [],                                    ['server-momentum', 'server-lr', 'proximal-mu']],
    ['fedavgm',  ['server-momentum', 'server-lr'],      ['proximal-mu']],
    ['fedprox',  ['proximal-mu'],                       ['server-momentum', 'server-lr']],
    ['fednovam', ['server-momentum', 'server-lr'],      ['proximal-mu']],
  ])(
    'strategy %s keeps %p and drops %p',
    (strategy, kept, dropped) => {
      const result = filterRunConfig(FULL_CFG, strategy)
      kept.forEach((k) => expect(result).toHaveProperty(k))
      dropped.forEach((k) => expect(result).not.toHaveProperty(k))
    },
  )

  it('preserves all non-gated keys regardless of strategy', () => {
    const result = filterRunConfig(FULL_CFG, 'fedavg')
    const nonGated = [
      'model', 'aggregation', 'num-server-rounds', 'local-epochs',
      'fraction-train', 'min-train-nodes', 'min-available-nodes',
      'client-lr', 'client-momentum', 'client-weight-decay',
      'batch-size', 'optimizer',
      'straggler-mode', 'straggler-target', 'straggler-tolerance',
      'straggler-drop-tolerance', 'straggler-max-dropped',
      'straggler-min-chunk', 'straggler-min-epochs',
    ]
    nonGated.forEach((k) => expect(result).toHaveProperty(k))
  })

  it('does not mutate input', () => {
    const cfg = { ...FULL_CFG }
    const snapshot = JSON.parse(JSON.stringify(cfg))
    filterRunConfig(cfg, 'fedavg')
    expect(cfg).toEqual(snapshot)
  })

  it('regression: run #11 leak — drops proximal-mu when strategy=fedavgm', () => {
    const run11 = {
      model: 'se_resnet',
      'client-lr': 0.03,
      optimizer: 'sgd',
      'server-lr': 1,
      'batch-size': 64,
      aggregation: 'fedavgm',
      'proximal-mu': 0.0005,
      'local-epochs': 3,
      'fraction-train': 1,
      'client-momentum': 0.9,
      'min-train-nodes': 10,
      'server-momentum': 0.5,
      'num-server-rounds': 80,
    }
    const cleaned = filterRunConfig(run11, 'fedavgm')
    expect(cleaned).not.toHaveProperty('proximal-mu')
    expect(cleaned['server-momentum']).toBe(0.5)
  })
})

describe('STRATEGY_SPECIFIC_KEYS x STRATEGY_GATED_KEYS consistency', () => {
  it('every strategy-specific key is also marked as gated', () => {
    for (const s of ALL_STRATEGIES) {
      for (const k of STRATEGY_SPECIFIC_KEYS[s]) {
        expect(STRATEGY_GATED_KEYS).toContain(k)
      }
    }
  })

  it('every gated key belongs to at least one strategy', () => {
    const allInUse = new Set<string>()
    for (const s of ALL_STRATEGIES) {
      STRATEGY_SPECIFIC_KEYS[s].forEach((k) => allInUse.add(k))
    }
    for (const k of STRATEGY_GATED_KEYS) {
      expect(allInUse).toContain(k)
    }
  })
})
