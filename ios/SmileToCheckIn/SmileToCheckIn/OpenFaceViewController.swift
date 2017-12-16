//
//  OpenFaceViewController.swift
//  SmileToCheckIn
//
//  Created by 鈴木治 on 2017/12/16.
//  Copyright © 2017年 Osamu Suzuki. All rights reserved.
//

import UIKit
import CoreML
import AVFoundation
import Vision
import Accelerate

class OpenFaceViewController: UIViewController {

    @IBOutlet weak var imageView: UIImageView!
    
    var model: OpenFace!
    var session = AVCaptureSession()
    var requests = [VNRequest]()
    var currentPixelBuffer: CVPixelBuffer?
    var count = 0
    var currentLabelRect = [CGRect]()
    var labelsArray: [String]?
    var repsMatrix: Matrix<Double>?
    var shouldGemEmbedding = 1
    var start = CACurrentMediaTime()
    var end = CACurrentMediaTime()
    
    
    lazy var MLRequest: VNCoreMLRequest = {
        // Load the ML model through its generated class and create a Vision request for it.
        do {
            let model = try VNCoreMLModel(for: OpenFace().model)
            return VNCoreMLRequest(model: model, completionHandler: self.genEmbeddingsHandler)
        } catch {
            fatalError("can't load Vision ML model: \(error)")
        }
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()

        
        self.imageView.image = #imageLiteral(resourceName: "carell_aligned")
        
        
        //UIImage -> CVPixelBuffer
        let pixelBuffer = pixelBufferFromImage(image: #imageLiteral(resourceName: "carell_aligned"))
        do {
            self.currentPixelBuffer = pixelBuffer
            
            
            self.requestML()
            
        } catch {
            print(error)
        }
    }

    func requestML() {
        
        let cropAndResizeFaceQueue = DispatchQueue(label: "com.wangderland.cropAndResizeQueue", qos: .userInteractive)
        
        cropAndResizeFaceQueue.async {
            guard let pixelBuffer = self.currentPixelBuffer else { return }
//            let boundingRect = region.boundingBox
//            let boundingRect = CGRect(x: 0, y: 0, width: 96, height: 96)
//            let x = boundingRect.minX * CGFloat(CVPixelBufferGetWidth(pixelBuffer))
//            let w = boundingRect.width * CGFloat(CVPixelBufferGetWidth(pixelBuffer))
//            let h = boundingRect.height * CGFloat(CVPixelBufferGetHeight(pixelBuffer))
//            let y = CGFloat(CVPixelBufferGetHeight(pixelBuffer)) * (1 - boundingRect.minY) - h
//            let scaledRect = CGRect(x: x, y: y, width: w, height: h)
//            guard let croppedPixelBuffer = self.cropFace(imageBuffer: pixelBuffer, region: scaledRect) else { return }
            let croppedPixelBuffer = pixelBuffer
            let MLRequestHandler = VNImageRequestHandler(cvPixelBuffer: croppedPixelBuffer, orientation: CGImagePropertyOrientation(rawValue: 1)!, options: [:])
            do {
//                let scaledRect = self.scale(rect: boundingRect, view: self.preview)
//                self.currentLabelRect.append(CGRect(x: scaledRect.minX, y: scaledRect.minY - 60, width: 200, height: 50))
                try MLRequestHandler.perform([self.MLRequest])
            } catch {
                print(error)
            }
        }
        
    }
    
    func scale(rect: CGRect, view: UIView) -> CGRect {
        let x = rect.minX * view.frame.size.width
        let w = rect.width * view.frame.size.width
        let h = rect.height * view.frame.size.height
        let y = view.frame.size.height * (1 - rect.minY) - h
        return CGRect(x: x, y: y, width: w, height: h)
    }
    
    func buffer2Array<T>(length: Int, data: UnsafeMutableRawPointer, _: T.Type) -> [T] {
        let ptr = data.bindMemory(to: T.self, capacity: length)
        let buffer = UnsafeBufferPointer(start: ptr, count: length)
        return Array(buffer)
    }
    
    func genEmbeddingsHandler(request: VNRequest, error: Error?) {
        
        guard let observations = request.results as? [ VNCoreMLFeatureValueObservation] else { return }
        observations.forEach { observe in
            self.start = CACurrentMediaTime()
            guard let emb = observe.featureValue.multiArrayValue else { return }
            print(emb)
            let doubleValueEmb = buffer2Array(length: emb.count, data: emb.dataPointer, Double.self)
            print(doubleValueEmb)
            
//            guard let repsMatrix = self.repsMatrix else { return }
//            let embMatrix = Matrix(Array(repeating: doubleValueEmb, count: repsMatrix.rows))
//            let diff = repsMatrix - embMatrix
//            let squredDiff = myPow(diff, 2)
//            let l2 = sum(squredDiff, axies:.row)
//            let grid = l2.grid
//            let minVal = l2.grid.min()
//            var ans: String = "Unknown"
//            guard let minIdx = l2.grid.index(of: minVal!) else { return }
//
//
//
//            self.end = CACurrentMediaTime()
//            print("name: \(ans), distance: \(minVal ?? nil)")
//            print("Gem time", self.end - self.start)
            
            
        }
    }

}
