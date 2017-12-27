//
//  scaling.swift
//  SmileToCheckIn
//
//  Created by 鈴木治 on 2017/12/25.
//  Copyright © 2017年 Osamu Suzuki. All rights reserved.
//

import Foundation
import CoreML
import Accelerate

@objc(scaling) class scaling: NSObject, MLCustomLayer {
    required init(parameters: [String : Any]) throws {
        print(#function, parameters)
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws {
        print(#function, weights)
        
        // This layer does not have any learned weights. However, in the conversion
        // script we added some (random) weights anyway, just to see how this works.
        // Here you would copy those weights into a buffer (such as MTLBuffer).
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws
        -> [[NSNumber]] {
//            print(#function, inputShapes)
            // This layer does not modify the size of the data.
            return inputShapes
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
//        print(#function, inputs.count, outputs.count)
        print(#function, inputs[0].shape, outputs[0].shape)
        print(inputs[0], outputs[0])
        //TODO: fix
        
//        for i in 0..<inputs.count {
//            let input = inputs[i]
//            let output = outputs[i]
//            // NOTE: In a real app, you might need to handle different datatypes.
//            // We only support 32-bit floats for now.
//            assert(input.dataType == .float32)
//            assert(output.dataType == .float32)
//            assert(input.shape == output.shape)
//
//            for j in 0..<input.count {
//                let x = input[j].floatValue
//                let y = x / (1 + exp(-x))
//                output[j] = NSNumber(value: y)
//            }
//        }
        
    }
}
