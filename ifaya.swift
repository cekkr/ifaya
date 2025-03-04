import Foundation
import CoreML
import Accelerate

/**
 SwifTorch: Libreria Swift per iOS che estende CoreML
 offrendo un'interfaccia simile a PyTorch
 */

// MARK: - Tensore
public class Tensor {
    var data: [Float]
    var shape: [Int]
    var requiresGrad: Bool
    var grad: Tensor?
    
    public init(data: [Float], shape: [Int], requiresGrad: Bool = false) {
        self.data = data
        self.shape = shape
        self.requiresGrad = requiresGrad
        
        if requiresGrad {
            self.grad = Tensor(data: Array(repeating: 0.0, count: data.count), shape: shape)
        }
    }
    
    public convenience init(shape: [Int], requiresGrad: Bool = false) {
        let count = shape.reduce(1, *)
        self.init(data: Array(repeating: 0.0, count: count), shape: shape, requiresGrad: requiresGrad)
    }
    
    public static func randn(shape: [Int], requiresGrad: Bool = false) -> Tensor {
        let count = shape.reduce(1, *)
        let data = (0..<count).map { _ in Float.random(in: -1...1) }
        return Tensor(data: data, shape: shape, requiresGrad: requiresGrad)
    }
    
    public static func zeros(shape: [Int], requiresGrad: Bool = false) -> Tensor {
        let count = shape.reduce(1, *)
        return Tensor(data: Array(repeating: 0.0, count: count), shape: shape, requiresGrad: requiresGrad)
    }
    
    public static func ones(shape: [Int], requiresGrad: Bool = false) -> Tensor {
        let count = shape.reduce(1, *)
        return Tensor(data: Array(repeating: 1.0, count: count), shape: shape, requiresGrad: requiresGrad)
    }
    
    // Operazioni base
    public func add(_ other: Tensor) -> Tensor {
        precondition(self.shape == other.shape, "Shapes must match for addition")
        
        var resultData = [Float](repeating: 0, count: self.data.count)
        vDSP_vadd(self.data, 1, other.data, 1, &resultData, 1, vDSP_Length(self.data.count))
        
        let result = Tensor(data: resultData, shape: self.shape, requiresGrad: self.requiresGrad || other.requiresGrad)
        
        return result
    }
    
    public func sub(_ other: Tensor) -> Tensor {
        precondition(self.shape == other.shape, "Shapes must match for subtraction")
        
        var resultData = [Float](repeating: 0, count: self.data.count)
        vDSP_vsub(other.data, 1, self.data, 1, &resultData, 1, vDSP_Length(self.data.count))
        
        let result = Tensor(data: resultData, shape: self.shape, requiresGrad: self.requiresGrad || other.requiresGrad)
        
        return result
    }
    
    public func mul(_ other: Tensor) -> Tensor {
        precondition(self.shape == other.shape, "Shapes must match for element-wise multiplication")
        
        var resultData = [Float](repeating: 0, count: self.data.count)
        vDSP_vmul(self.data, 1, other.data, 1, &resultData, 1, vDSP_Length(self.data.count))
        
        let result = Tensor(data: resultData, shape: self.shape, requiresGrad: self.requiresGrad || other.requiresGrad)
        
        return result
    }
    
    public func matmul(_ other: Tensor) -> Tensor {
        precondition(self.shape.count >= 2 && other.shape.count >= 2, "Tensors must have at least 2 dimensions")
        precondition(self.shape[self.shape.count - 1] == other.shape[other.shape.count - 2], "Inner dimensions must match")
        
        let m = self.shape[self.shape.count - 2]
        let n = other.shape[other.shape.count - 1]
        let k = self.shape[self.shape.count - 1]
        
        var resultShape = self.shape
        resultShape[resultShape.count - 1] = n
        
        var resultData = [Float](repeating: 0, count: resultShape.reduce(1, *))
        
        // Esegue moltiplicazione matriciale con Accelerate
        let selfData = self.data
        let otherData = other.data
        
        var alpha: Float = 1.0
        var beta: Float = 0.0
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(m), Int32(n), Int32(k),
                    alpha,
                    selfData, Int32(k),
                    otherData, Int32(n),
                    beta,
                    &resultData, Int32(n))
        
        let result = Tensor(data: resultData, shape: resultShape, requiresGrad: self.requiresGrad || other.requiresGrad)
        
        return result
    }
    
    public func relu() -> Tensor {
        var resultData = self.data
        for i in 0..<resultData.count {
            resultData[i] = max(0, resultData[i])
        }
        
        let result = Tensor(data: resultData, shape: self.shape, requiresGrad: self.requiresGrad)
        
        return result
    }
    
    public func tanh() -> Tensor {
        var resultData = self.data
        vvtanhf(&resultData, self.data, [Int32(self.data.count)])
        
        let result = Tensor(data: resultData, shape: self.shape, requiresGrad: self.requiresGrad)
        
        return result
    }
    
    public func sigmoid() -> Tensor {
        var resultData = [Float](repeating: 0, count: self.data.count)
        
        for i in 0..<self.data.count {
            resultData[i] = 1.0 / (1.0 + exp(-self.data[i]))
        }
        
        let result = Tensor(data: resultData, shape: self.shape, requiresGrad: self.requiresGrad)
        
        return result
    }
    
    public func softmax() -> Tensor {
        var resultData = [Float](repeating: 0, count: self.data.count)
        
        let maxVal = self.data.max() ?? 0
        var expSum: Float = 0
        
        for i in 0..<self.data.count {
            let expVal = exp(self.data[i] - maxVal)
            resultData[i] = expVal
            expSum += expVal
        }
        
        for i in 0..<resultData.count {
            resultData[i] /= expSum
        }
        
        let result = Tensor(data: resultData, shape: self.shape, requiresGrad: self.requiresGrad)
        
        return result
    }
    
    public func sum() -> Tensor {
        var result: Float = 0
        vDSP_sve(self.data, 1, &result, vDSP_Length(self.data.count))
        
        return Tensor(data: [result], shape: [1], requiresGrad: self.requiresGrad)
    }
    
    public func reshape(_ newShape: [Int]) -> Tensor {
        precondition(self.data.count == newShape.reduce(1, *), "New shape must have the same number of elements")
        
        return Tensor(data: self.data, shape: newShape, requiresGrad: self.requiresGrad)
    }
}

// MARK: - Modulo Base
public protocol Module {
    func forward(_ input: Tensor) -> Tensor
    func parameters() -> [Tensor]
}

// MARK: - Livelli
public class Linear: Module {
    var weight: Tensor
    var bias: Tensor
    
    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true) {
        // Inizializzazione di He
        let stdv = sqrt(2.0 / Float(inFeatures))
        self.weight = Tensor.randn(shape: [inFeatures, outFeatures], requiresGrad: true)
        
        // Scala i pesi
        for i in 0..<self.weight.data.count {
            self.weight.data[i] *= stdv
        }
        
        if bias {
            self.bias = Tensor(shape: [1, outFeatures], requiresGrad: true)
        } else {
            self.bias = Tensor(shape: [1, outFeatures], requiresGrad: false)
        }
    }
    
    public func forward(_ input: Tensor) -> Tensor {
        let output = input.matmul(weight)
        return output.add(bias)
    }
    
    public func parameters() -> [Tensor] {
        return [weight, bias]
    }
}

public class ReLU: Module {
    public init() {}
    
    public func forward(_ input: Tensor) -> Tensor {
        return input.relu()
    }
    
    public func parameters() -> [Tensor] {
        return []
    }
}

public class Tanh: Module {
    public init() {}
    
    public func forward(_ input: Tensor) -> Tensor {
        return input.tanh()
    }
    
    public func parameters() -> [Tensor] {
        return []
    }
}

public class Sigmoid: Module {
    public init() {}
    
    public func forward(_ input: Tensor) -> Tensor {
        return input.sigmoid()
    }
    
    public func parameters() -> [Tensor] {
        return []
    }
}

public class GRUCell: Module {
    private let inputSize: Int
    private let hiddenSize: Int
    
    // Pesi per update gate
    private let wiz: Linear
    private let whz: Linear
    
    // Pesi per reset gate
    private let wir: Linear
    private let whr: Linear
    
    // Pesi per new gate
    private let win: Linear
    private let whn: Linear
    
    public init(inputSize: Int, hiddenSize: Int) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        
        // Inizializza i gate lineari
        self.wiz = Linear(inFeatures: inputSize, outFeatures: hiddenSize)
        self.whz = Linear(inFeatures: hiddenSize, outFeatures: hiddenSize)
        
        self.wir = Linear(inFeatures: inputSize, outFeatures: hiddenSize)
        self.whr = Linear(inFeatures: hiddenSize, outFeatures: hiddenSize)
        
        self.win = Linear(inFeatures: inputSize, outFeatures: hiddenSize)
        self.whn = Linear(inFeatures: hiddenSize, outFeatures: hiddenSize)
    }
    
    public func forward(_ input: Tensor, hidden: Tensor? = nil) -> Tensor {
        let batchSize = input.shape[0]
        let h = hidden ?? Tensor.zeros(shape: [batchSize, hiddenSize])
        
        // Update gate
        let z = self.wiz.forward(input).add(self.whz.forward(h)).sigmoid()
        
        // Reset gate
        let r = self.wir.forward(input).add(self.whr.forward(h)).sigmoid()
        
        // New gate
        let n = self.win.forward(input).add(self.whn.forward(r.mul(h))).tanh()
        
        // Output
        let onesTensor = Tensor.ones(shape: z.shape)
        let zComplement = onesTensor.sub(z)
        
        return z.mul(h).add(zComplement.mul(n))
    }
    
    public func parameters() -> [Tensor] {
        return wiz.parameters() + whz.parameters() +
               wir.parameters() + whr.parameters() +
               win.parameters() + whn.parameters()
    }
}

public class Sequential: Module {
    private var modules: [Module]
    
    public init(_ modules: Module...) {
        self.modules = modules
    }
    
    public func forward(_ input: Tensor) -> Tensor {
        var current = input
        
        for module in modules {
            current = module.forward(current)
        }
        
        return current
    }
    
    public func parameters() -> [Tensor] {
        return modules.flatMap { $0.parameters() }
    }
}

// MARK: - Funzioni di Perdita
public protocol Loss {
    func forward(_ predictions: Tensor, _ targets: Tensor) -> Tensor
}

public class MSELoss: Loss {
    public init() {}
    
    public func forward(_ predictions: Tensor, _ targets: Tensor) -> Tensor {
        precondition(predictions.shape == targets.shape, "Predictions and targets must have the same shape")
        
        let diff = predictions.sub(targets)
        let squared = diff.mul(diff)
        
        return squared.sum()
    }
}

public class CrossEntropyLoss: Loss {
    public init() {}
    
    public func forward(_ predictions: Tensor, _ targets: Tensor) -> Tensor {
        let logSoftmax = predictions.softmax()
        var loss: Float = 0.0
        
        // Molto semplificato - in una implementazione reale si dovrebbe gestire meglio
        for i in 0..<targets.data.count {
            let targetClass = Int(targets.data[i])
            if targetClass < logSoftmax.data.count {
                loss -= log(max(logSoftmax.data[targetClass], 1e-10))
            }
        }
        
        return Tensor(data: [loss], shape: [1], requiresGrad: true)
    }
}

// MARK: - Ottimizzatori
public protocol Optimizer {
    func zeroGrad()
    func step()
}

public class SGD: Optimizer {
    private let parameters: [Tensor]
    private let learningRate: Float
    private let momentum: Float
    private var velocities: [[Float]]
    
    public init(parameters: [Tensor], learningRate: Float = 0.01, momentum: Float = 0.0) {
        self.parameters = parameters
        self.learningRate = learningRate
        self.momentum = momentum
        
        self.velocities = parameters.map { param in
            return [Float](repeating: 0.0, count: param.data.count)
        }
    }
    
    public func zeroGrad() {
        for param in parameters where param.requiresGrad {
            if let grad = param.grad {
                for i in 0..<grad.data.count {
                    grad.data[i] = 0.0
                }
            }
        }
    }
    
    public func step() {
        for (i, param) in parameters.enumerated() where param.requiresGrad {
            if let grad = param.grad {
                var velocity = velocities[i]
                
                for j in 0..<param.data.count {
                    velocity[j] = momentum * velocity[j] - learningRate * grad.data[j]
                    param.data[j] += velocity[j]
                }
                
                velocities[i] = velocity
            }
        }
    }
}

public class Adam: Optimizer {
    private let parameters: [Tensor]
    private let learningRate: Float
    private let beta1: Float
    private let beta2: Float
    private let epsilon: Float
    
    private var m: [[Float]]
    private var v: [[Float]]
    private var t: Int = 0
    
    public init(parameters: [Tensor], learningRate: Float = 0.001, beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8) {
        self.parameters = parameters
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = parameters.map { param in
            return [Float](repeating: 0.0, count: param.data.count)
        }
        
        self.v = parameters.map { param in
            return [Float](repeating: 0.0, count: param.data.count)
        }
    }
    
    public func zeroGrad() {
        for param in parameters where param.requiresGrad {
            if let grad = param.grad {
                for i in 0..<grad.data.count {
                    grad.data[i] = 0.0
                }
            }
        }
    }
    
    public func step() {
        t += 1
        let correctedLR = learningRate * sqrt(1.0 - pow(beta2, Float(t))) / (1.0 - pow(beta1, Float(t)))
        
        for (i, param) in parameters.enumerated() where param.requiresGrad {
            if let grad = param.grad {
                var mi = m[i]
                var vi = v[i]
                
                for j in 0..<param.data.count {
                    let g = grad.data[j]
                    
                    mi[j] = beta1 * mi[j] + (1.0 - beta1) * g
                    vi[j] = beta2 * vi[j] + (1.0 - beta2) * g * g
                    
                    param.data[j] -= correctedLR * mi[j] / (sqrt(vi[j]) + epsilon)
                }
                
                m[i] = mi
                v[i] = vi
            }
        }
    }
}

// MARK: - CoreML Integration
public class MLModelConverter {
    public static func convertToMLModel(model: Module, inputShape: [Int], outputShape: [Int]) -> MLModel? {
        // Questa è una versione di base - in una implementazione reale,
        // qui si convertirebbe il modello SwifTorch in un CoreML .mlmodel
        // usando CoreML Tools o una procedura di conversione personalizzata
        
        // Per ora, ritorniamo nil
        return nil
    }
}