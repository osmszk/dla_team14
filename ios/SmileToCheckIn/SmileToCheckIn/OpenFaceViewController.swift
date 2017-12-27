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

@available(iOS 11.2, *)
class OpenFaceViewController: UIViewController {

    @IBOutlet weak var imageView: UIImageView!
    
    var model: FaceNetModel!
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
            let model = try VNCoreMLModel(for: FaceNetModel().model)
            return VNCoreMLRequest(model: model, completionHandler: self.genEmbeddingsHandler)
        } catch {
            fatalError("can't load Vision ML model: \(error)")
        }
    }()
    
    lazy var image: UIImage = {
        let image = #imageLiteral(resourceName: "clapton-2")
        //#imageLiteral(resourceName: "clapton-1")
        //#imageLiteral(resourceName: "clapton-2")
        //#imageLiteral(resourceName: "lennon-2")
        
        let newWidth:CGFloat = 160
        let scale = newWidth / image.size.width
        let newHeight = image.size.height * scale
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newWidth, height: newHeight), true, 3.0)
        image.draw(in: CGRect(x:0, y:0, width:newWidth, height:newHeight))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return newImage!
    }()
    let csvName: String = "lennon2"
    //clapton1
    //lennon2
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        print(NSHomeDirectory())
        
        readDataFromCSV()
        
        let image = self.image
        self.imageView.image = image
        
        startFaceDetection()
        
        //UIImage -> CVPixelBuffer
        let pixelBuffer = pixelBufferFromImage(image: self.image)
        let requestOptions:[VNImageOption : Any] = [:]
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: CGImagePropertyOrientation(rawValue: 1)!, options: requestOptions)
        do {
            self.currentPixelBuffer = pixelBuffer
            try imageRequestHandler.perform(self.requests)
            self.count += 1
        } catch {
            print(error)
        }
    }
    
    func startFaceDetection() {
        let faceRequest = VNDetectFaceRectanglesRequest(completionHandler: self.detectFaceHandler)
        self.requests = [faceRequest]
    }
    
    func detectFaceHandler(request: VNRequest, error: Error?) {
        print("detectFaceHandler!!")
        
        guard let observations = request.results as? [VNFaceObservation] else {
            print("no result")
            return
        }
        DispatchQueue.main.async() {
            self.imageView.layer.sublayers?.removeSubrange(1...)
            for region in observations {
                self.highlightFace(faceObservation: region)
            }
        }
        
//        if (self.shouldGemEmbedding % 2 == 0) {
//            self.shouldGemEmbedding = 1
//            self.currentLabelRect = []
//        } else {
//            self.shouldGemEmbedding = self.shouldGemEmbedding + 1
//            return
//        }
        let cropAndResizeFaceQueue = DispatchQueue(label: "osuzuki.SmileToCheckIn", qos: .userInteractive)
        for region in observations {
            cropAndResizeFaceQueue.async {
                guard let pixelBuffer = self.currentPixelBuffer else { return }
                let boundingRect = region.boundingBox
                let x = boundingRect.minX * CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                let w = boundingRect.width * CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                let h = boundingRect.height * CGFloat(CVPixelBufferGetHeight(pixelBuffer))
                let y = CGFloat(CVPixelBufferGetHeight(pixelBuffer)) * (1 - boundingRect.minY) - h
                let scaledRect = CGRect(x: x, y: y, width: w, height: h)
                print("boundingRect:\(boundingRect) scaledRect:\(scaledRect)")
                guard let croppedPixelBuffer = self.cropFace(imageBuffer: pixelBuffer, region: scaledRect) else { return }
//                self.showImageAsTest(name: "croppedPixelBuffer", pixelBuffer: croppedPixelBuffer)
                
                let MLRequestHandler = VNImageRequestHandler(cvPixelBuffer: croppedPixelBuffer, orientation: CGImagePropertyOrientation(rawValue: 1)!, options: [:])
                do {
//                    let scaledRect = self.scale(rect: boundingRect, view: self.imageView)
//                    self.currentLabelRect.append(CGRect(x: scaledRect.minX, y: scaledRect.minY - 60, width: 200, height: 50))
                    try MLRequestHandler.perform([self.MLRequest])
                } catch {
                    print(error)
                }
            }
        }
    }
    
    func showImageAsTest(name: String, pixelBuffer: CVPixelBuffer) {
        DispatchQueue.main.async {
            if let image = UIImage(pixelBuffer: pixelBuffer) {
                print("showImageAsTest size:\(image.size)")
                let imageView = UIImageView(frame: CGRect(origin: CGPoint.zero, size: image.size))
                imageView.image = image
                self.view.addSubview(imageView)
            }
        }
    }
    
    func cropFace(imageBuffer: CVPixelBuffer, region: CGRect) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(imageBuffer, .readOnly)
        let baseAddress = CVPixelBufferGetBaseAddress(imageBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer)
        // calculate start position
        let bytesPerPixel = 4
        let startAddress = baseAddress?.advanced(by: Int(region.minY) * bytesPerRow + Int(region.minX) * bytesPerPixel)
        var croppedImageBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreateWithBytes(kCFAllocatorDefault,
                                                  Int(region.width),
                                                  Int(region.height),
                                                  kCVPixelFormatType_32BGRA,
                                                  startAddress!,
                                                  bytesPerRow,
                                                  nil,
                                                  nil,
                                                  nil,
                                                  &croppedImageBuffer)
        CVPixelBufferUnlockBaseAddress(imageBuffer, .readOnly)
        if (status != 0) {
            print("CVPixelBufferCreate Error: ", status)
        }
        return croppedImageBuffer
    }
    
    func highlightFace(faceObservation: VNFaceObservation) {
        
        let boundingRect = faceObservation.boundingBox
        print("highlightFace! \(boundingRect)")
        let x = boundingRect.minX * imageView.frame.size.width
        let w = boundingRect.width * imageView.frame.size.width
        let h = boundingRect.height * imageView.frame.size.height
        let y = imageView.frame.size.height * (1 - boundingRect.minY) - h
        let rect = CGRect(x: x, y: y, width: w, height: h)
        
        let outline = CAShapeLayer()
        outline.frame = rect
        outline.borderWidth = 1.0
        outline.borderColor = UIColor.red.cgColor
        imageView.layer.addSublayer(outline)
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
//            print(emb)
            let doubleValueEmb = buffer2Array(length: emb.count, data: emb.dataPointer, Double.self)
//            print(doubleValueEmb)
//            print("row:\(doubleValueEmb.rows) col:\(doubleValueEmb.columns) grid:\(doubleValueEmb.grid)")
            
            guard let repsMatrix = self.repsMatrix else { return }
            print("repsMatrix \(self.csvName)")
            print(repsMatrix.description)
            let embMatrix = Matrix(Array(repeating: doubleValueEmb, count: repsMatrix.rows))
            print("embMatrix")
            print(embMatrix.description)
            
            let diff = repsMatrix - embMatrix
//            let result2 = mul(diff, y: transpose(diff))
//            print(result2.description)
            
            print("diff")
            let squredDiff = myPow(diff, 2)
//            print(squredDiff.description)
            let l2 = sum(squredDiff, axies:.row)
            print("squared L2 distance !!!!")
            print(l2.description)
            
            
        }
    }
    
    func readDataFromCSV() {
        
        guard let repsPath = Bundle.main.path(forResource: self.csvName, ofType: "csv") else { return }
        
        let reps = try! String(contentsOfFile: repsPath, encoding: String.Encoding.utf8)
        
        self.start = CACurrentMediaTime()
        let repsArray: [[Double]] = reps
            .components(separatedBy: "\r")
            .filter{ $0.count > 0 }
            .map{
                //                print($0.components(separatedBy: ","))
                return $0.replacingOccurrences(of: " ", with: "").components(separatedBy: ",").map{ Double($0)! }
        }
        print("Done Reps")
        let repsMatrix = Matrix(repsArray)
        
        self.repsMatrix = repsMatrix
        print("rows:\(repsMatrix.rows)")
        print("columns:\(repsMatrix.columns)")
        print("grid count:\(repsMatrix.grid.count)  valu:\(repsMatrix.grid) ")
        
        self.end = CACurrentMediaTime()
        print("Done Import data:", self.end - self.start)
    }

}
