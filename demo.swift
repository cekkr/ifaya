import Foundation
import UIKit

// Esempio di utilizzo della libreria SwifTorch

// MARK: - 1. Creazione e manipolazione di tensori
func demoTensors() {
    print("=== DEMO TENSORI ===")
    
    // Creazione tensori
    let a = Tensor(data: [1, 2, 3, 4], shape: [2, 2], requiresGrad: true)
    let b = Tensor(data: [5, 6, 7, 8], shape: [2, 2], requiresGrad: true)
    
    print("Tensore A:")
    print(a.data)
    print("Shape: \(a.shape)")
    
    // Operazioni sui tensori
    let c = a.add(b)
    print("A + B:")
    print(c.data)
    
    let d = a.matmul(b)
    print("A @ B (prodotto matriciale):")
    print(d.data)
    
    // Tensori casuali
    let random = Tensor.randn(shape: [2, 3])
    print("Tensore casuale (2x3):")
    print(random.data)
    
    // Applicazione di funzioni di attivazione
    let activated = a.relu()
    print("ReLU(A):")
    print(activated.data)
}

// MARK: - 2. Definizione di un modello semplice (come torch.nn.Sequential)
func demoSimpleModel() {
    print("\n=== DEMO MODELLO SEMPLICE ===")
    
    // Creiamo un modello simile a quello che si farebbe con torch
    let model = Sequential(
        Linear(inFeatures: 4, outFeatures: 8),
        ReLU(),
        Linear(inFeatures: 8, outFeatures: 2)
    )
    
    // Preparazione input
    let batch = Tensor.randn(shape: [10, 4])  // Batch di 10 esempi, 4 features
    
    // Forward pass
    let output = model.forward(batch)
    print("Output shape: \(output.shape)")
    print("Output dei primi risultati: \(output.data.prefix(4))")
    
    // Estrazione dei parametri
    let params = model.parameters()
    print("Numero di tensori parametri: \(params.count)")
    print("Totale parametri: \(params.map { $0.data.count }.reduce(0, +))")
}

// MARK: - 3. Creazione di rete ricorrente GRU
func demoGRUModel() {
    print("\n=== DEMO GRU ===")
    
    // Dimensioni
    let batchSize = 3
    let sequenceLength = 5
    let inputSize = 4
    let hiddenSize = 8
    
    // Creazione cella GRU
    let gru = GRUCell(inputSize: inputSize, hiddenSize: hiddenSize)
    
    // Input simulato (batch, features)
    let input = Tensor.randn(shape: [batchSize, inputSize])
    
    // Stato nascosto iniziale
    var hidden = Tensor.zeros(shape: [batchSize, hiddenSize])
    
    // Simuliamo una sequenza
    print("Simulazione sequenza di lunghezza \(sequenceLength):")
    for i in 0..<sequenceLength {
        hidden = gru.forward(input, hidden: hidden)
        print("Step \(i+1) - Primi valori hidden: \(Array(hidden.data.prefix(3)))")
    }
}

// MARK: - 4. Training con ottimizzatore
func demoTraining() {
    print("\n=== DEMO TRAINING ===")
    
    // Dati di esempio
    let xTrain = Tensor.randn(shape: [100, 2])  // 100 esempi, 2 features
    
    // Target di esempio: somma delle due features
    var yTrainData: [Float] = []
    for i in stride(from: 0, to: xTrain.data.count, by: 2) {
        yTrainData.append(xTrain.data[i] + xTrain.data[i+1])
    }
    let yTrain = Tensor(data: yTrainData, shape: [100, 1])
    
    // Creazione modello
    let model = Sequential(
        Linear(inFeatures: 2, outFeatures: 8),
        ReLU(),
        Linear(inFeatures: 8, outFeatures: 1)
    )
    
    // Funzione di perdita
    let criterion = MSELoss()
    
    // Ottimizzatore
    let optimizer = Adam(parameters: model.parameters(), learningRate: 0.01)
    
    // Ciclo di training
    let epochs = 50
    for epoch in 0..<epochs {
        // Forward pass
        let predictions = model.forward(xTrain)
        let loss = criterion.forward(predictions, yTrain)
        
        // Backward pass (in una versione completa qui calcoleremmo i gradienti)
        // In questo esempio, simuliamo solo l'aggiornamento dei pesi
        
        // Azzera gradienti
        optimizer.zeroGrad()
        
        // Aggiorna parametri
        optimizer.step()
        
        if epoch % 10 == 0 {
            print("Epoca \(epoch) - Loss: \(loss.data[0])")
        }
    }
    
    // Valuta il modello
    let finalPredictions = model.forward(xTrain)
    let finalLoss = criterion.forward(finalPredictions, yTrain)
    print("Loss finale: \(finalLoss.data[0])")
}

// MARK: - 5. Conversione in CoreML
func demoMLModelConversion() {
    print("\n=== DEMO CONVERSIONE IN COREML ===")
    
    // Creazione modello
    let model = Sequential(
        Linear(inFeatures: 4, outFeatures: 8),
        ReLU(),
        Linear(inFeatures: 8, outFeatures: 1)
    )
    
    // Conversione
    if let mlModel = MLModelConverter.convertToMLModel(model: model, inputShape: [1, 4], outputShape: [1, 1]) {
        print("Modello convertito con successo!")
        // In una versione reale, qui salveremmo il modello o lo utilizzeremmo
    } else {
        print("Nota: La conversione in CoreML richiederebbe ulteriori implementazioni")
        print("Per completare questa funzionalità è necessario utilizzare coremltools")
    }
}

// Main demo
func runSwifTorchDemo() {
    print("DIMOSTRAZIONE SWIFTORCH")
    print("=======================")
    
    demoTensors()
    demoSimpleModel()
    demoGRUModel()
    demoTraining()
    demoMLModelConversion()
}

// Esegui la demo
runSwifTorchDemo()