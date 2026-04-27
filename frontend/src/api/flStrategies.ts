export type StrategyId = 'fedavg' | 'fedavgm' | 'fedprox' | 'fednovam'

export type FLParams = {
  // common
  'num-server-rounds': number
  'local-epochs': number
  'fraction-train': number
  'min-train-nodes': number
  'min-available-nodes': number
  // fedavgm / fednovam
  'server-momentum': number
  'server-lr': number
  // fedprox
  'proximal-mu': number
}

export const FL_STRATEGIES: { id: StrategyId; label: string }[] = [
  { id: 'fedavg', label: 'FedAvg' },
  { id: 'fedavgm', label: 'FedAvgM' },
  { id: 'fedprox', label: 'FedProx' },
  { id: 'fednovam', label: 'FedNovaM' },
]

export const COMMON_FL_KEYS: (keyof FLParams)[] = [
  'num-server-rounds',
  'local-epochs',
  'fraction-train',
  'min-train-nodes',
  'min-available-nodes',
]

export const STRATEGY_SPECIFIC_KEYS: Record<StrategyId, (keyof FLParams)[]> = {
  fedavg: [],
  fedavgm: ['server-momentum', 'server-lr'],
  fedprox: ['proximal-mu'],
  fednovam: ['server-momentum', 'server-lr'],
}

export const FL_PARAM_LABELS: Record<keyof FLParams, string> = {
  'num-server-rounds': 'Server rounds',
  'local-epochs': 'Local epochs',
  'fraction-train': 'Fraction of clients per round',
  'min-train-nodes': 'Min training nodes',
  'min-available-nodes': 'Min available nodes',
  'server-momentum': 'Server momentum',
  'server-lr': 'Server learning rate',
  'proximal-mu': 'Proximal μ',
}

export const FL_PARAM_RANGES: Record<
  keyof FLParams,
  { min: number; max: number; step: number }
> = {
  'num-server-rounds': { min: 1, max: 200, step: 1 },
  'local-epochs': { min: 1, max: 20, step: 1 },
  'fraction-train': { min: 0, max: 1, step: 0.05 },
  'min-train-nodes': { min: 1, max: 50, step: 1 },
  'min-available-nodes': { min: 1, max: 50, step: 1 },
  'server-momentum': { min: 0, max: 0.99, step: 0.01 },
  'server-lr': { min: 0, max: 2, step: 0.01 },
  'proximal-mu': { min: 0, max: 0.1, step: 0.0001 },
}

export const FL_PARAM_DEFAULTS: FLParams = {
  'num-server-rounds': 80,
  'local-epochs': 2,
  'fraction-train': 1.0,
  'min-train-nodes': 10,
  'min-available-nodes': 10,
  'server-momentum': 0.5,
  'server-lr': 1.0,
  'proximal-mu': 0.0005,
}
