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
    
    lazy var mlRequest: VNCoreMLRequest = {
        // Load the ML model through its generated class and create a Vision request for it.
        do {
            let model = try VNCoreMLModel(for: FaceNet7().model)
            return VNCoreMLRequest(model: model, completionHandler: self.genEmbeddingsHandler)
        } catch {
            fatalError("can't load Vision ML model: \(error)")
        }
    }()
    
    lazy var clapton1Image: UIImage = {
        return resize(image: #imageLiteral(resourceName: "clapton-1_aligned"))
    }()
    lazy var clapton2Image: UIImage = {
        return resize(image: #imageLiteral(resourceName: "clapton-2_aligned"))
    }()
    lazy var lennon1Image: UIImage = {
        return resize(image: #imageLiteral(resourceName: "lennon-1_aligned"))
    }()
    lazy var lennon2Image: UIImage = {
        return resize(image: #imageLiteral(resourceName: "lennon-2_aligned"))
    }()
    
    var matrixDic: [String : Matrix<Double>] = [:]
    var results: [[Double]] = []
    var normlizedDic: [String : Matrix<Double>] = [:]
    //key-> taniai1,taniai2,takemoto1,takemoto2
    
    var imageNameNumber: Int = 1
    let csvName: String = "takemoto2"
    //clapton1
    //lennon1
    //lennon2
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        print(NSHomeDirectory())
        
//        readDataFromCSV()
        
        let imageName = "taniai\(imageNameNumber).png"
        let image = UIImage(named: imageName)!
        self.imageView.image = image
        
        startFaceDetection()
        
        requestML(name: imageName, skipDetect: false)
    }
    
    func requestML(name: String , skipDetect: Bool = true) {
        print(#function, name, "--------------------------------------------------------")
        let image = UIImage(named: name)!
        self.requestML(image: image, skipDetect: skipDetect)
    }
    
    func requestML(image: UIImage, skipDetect: Bool = true) {
        if skipDetect {
            //顔検出はスキップする
            let MLRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBufferFromImage(image: image))
            do {
                try MLRequestHandler.perform([self.mlRequest])
            } catch {
                print(error)
            }
            return
        }
        
        let pixelBuffer = pixelBufferFromImage(image: image)
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        do {
            self.currentPixelBuffer = pixelBuffer
            try imageRequestHandler.perform(self.requests)
            self.count += 1
        } catch {
            print(error)
        }
    }
    
    func resize(image: UIImage) -> UIImage {
        let newWidth:CGFloat = 160
        let scale = newWidth / image.size.width
        let newHeight = image.size.height * scale
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newWidth, height: newHeight), true, 3.0)
        image.draw(in: CGRect(x:0, y:0, width:newWidth, height:newHeight))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage!
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
//        DispatchQueue.main.async() {
//            self.imageView.layer.sublayers?.removeSubrange(1...)
//            for region in observations {
//                self.highlightFace(faceObservation: region)
//            }
//        }
        
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
//                print("boundingRect:\(boundingRect) scaledRect:\(scaledRect)")
                guard let croppedPixelBuffer = self.cropFace(imageBuffer: pixelBuffer, region: scaledRect) else { return }
                
                let multiArray = self.toMultiArrayFromPixelBuffer(pixelBuffer: croppedPixelBuffer)
                print("multiArray shape",multiArray.shape)

                self.showImageAsTest(name: "croppedPixelBuffer", image: multiArray.transposed([2,0,1]).image(offset: 0, scale: 1)!)

                let prewhitened = self.prewhiten(multiArray).transposed([2,0,1]) //shape変更[w,h,c] -> [c,w,h]
                print("prewhitened shape",prewhitened.shape)
//                print("prewhitened",prewhitened)

                guard let uiImage = prewhitened.image(offset: 0.0, scale: 1.0) else {
                    print("uiImage is nil")
                    return
                }

                guard let ciImage = CIImage(image: uiImage) else {
                    print("ciImage is nil")
                    return
                }
                
                let MLRequestHandler = VNImageRequestHandler(ciImage: ciImage, options: [:])
//                let MLRequestHandler = VNImageRequestHandler(cvPixelBuffer: croppedPixelBuffer, options: [:])
                do {
//                    let scaledRect = self.scale(rect: boundingRect, view: self.imageView)
//                    self.currentLabelRect.append(CGRect(x: scaledRect.minX, y: scaledRect.minY - 60, width: 200, height: 50))
                    try MLRequestHandler.perform([self.mlRequest])
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
    
    func showImageAsTest(name: String, image: UIImage) {
        DispatchQueue.main.async {
            print("showImageAsTest size:\(image.size)")
            let imageView = UIImageView(frame: CGRect(origin: CGPoint.zero, size: image.size))
            imageView.image = image
            self.view.addSubview(imageView)
        }
    }
    
    func setImage(image: UIImage) {
        DispatchQueue.main.async {
            self.imageView.image = image
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
    
    func toMultiArrayFromPixelBuffer(pixelBuffer: CVPixelBuffer) -> MultiArray<Double>{
        print(#function,"")
        
        ////https://stackoverflow.com/questions/8072208/how-to-turn-a-cvpixelbuffer-into-a-uiimage
        
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0));
        
        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        let r = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        print("w,h,r",w,h,r)
        
        let channel = 3
        var m = MultiArray<Double>(shape: [h, w, channel])
        print(m.shape)
        if let buffer = CVPixelBufferGetBaseAddress(pixelBuffer) {
            
            let maxY = h
            
            print("A R G B")
            var i: Int = 0
            for y in 0..<maxY {
                for x in 0..<w {
                    let offset = 4*x + y*r
                    
                    //https://stackoverflow.com/questions/39548344/getting-pixel-color-from-an-image-using-cgpoint-in-swift-3
//                    let a = buffer.load(fromByteOffset: offset+3, as: UInt8.self)
                    let r = buffer.load(fromByteOffset: offset+2, as: UInt8.self)
                    let g = buffer.load(fromByteOffset: offset+1, as: UInt8.self)
                    let b = buffer.load(fromByteOffset: offset, as: UInt8.self)
//                    print("offset:",offset,"ARGB:",a,r,g,b)
                    m[i] = Double(r)
                    i += 1
                    m[i] = Double(g)
                    i += 1
                    m[i] = Double(b)
                    i += 1
                }
            }
        }
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue:1))
        return m
    }
    
    func highlightFace(faceObservation: VNFaceObservation) {
        
        let boundingRect = faceObservation.boundingBox
//        print("highlightFace! \(boundingRect)")
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
        guard let observations = request.results as? [ VNCoreMLFeatureValueObservation] else {
            return
        }
        observations.forEach { observe in
            self.start = CACurrentMediaTime()
            guard let emb = observe.featureValue.multiArrayValue else {
                return
            }
//            print(emb)
            let doubleValueEmb = buffer2Array(length: emb.count, data: emb.dataPointer, Double.self)
            print(doubleValueEmb)
            
            self.results.append(doubleValueEmb)
            
//            print("row:\(doubleValueEmb.rows) col:\(doubleValueEmb.columns) grid:\(doubleValueEmb.grid)")
            
            let embMatrix = Matrix(Array(repeating: doubleValueEmb, count: 1))
            var imageName = "taniai\(self.imageNameNumber).png"
            if (self.matrixDic[imageName] == nil) {
                self.matrixDic[imageName] = embMatrix
                self.imageNameNumber += 1
                imageName = "taniai\(self.imageNameNumber).png"
                if let img = UIImage(named: imageName) {
                    self.setImage(image: img)
                    self.requestML(name: imageName, skipDetect: false)
                    return
                } else {
                    print("*******[result taniai]*******", self.results.count, self.results[0].count)
//                    print(self.results)
                    
                    let l1 = self.l2Normalize(concat:self.results)
                    self.normlizedDic["taniai"] = l1
                    self.imageNameNumber = 1
                    
                    self.results.removeAll()

                    //next person
                    imageName = "takemoto\(self.imageNameNumber).png"
                    if let img = UIImage(named: imageName) {
                        self.setImage(image: img)
                        self.requestML(name: imageName, skipDetect: false)
                        return
                    }
                }
            }
            
            imageName = "takemoto\(self.imageNameNumber).png"

            if (self.matrixDic[imageName] == nil) {
                self.matrixDic[imageName] = embMatrix
                self.imageNameNumber += 1
                imageName = "takemoto\(self.imageNameNumber).png"
                if let img = UIImage(named: imageName) {
                    self.setImage(image: img)
                    self.requestML(name: imageName, skipDetect: false)
                    return
                } else {
                    print("*******[result takemoto]*******", self.results.count, self.results[0].count)
                    
                    let l2 = self.l2Normalize(concat:self.results)
                    self.normlizedDic["takemoto"] = l2
                    
                }
            }
//
//            self.diff()
            self.diffNormalized()
        }
    }
    
    func l2Normalize(concat: [[Double]]) -> Matrix<Double> {
        let m = Matrix<Double>(concat)
        
        let sq = myPow(m, 2)
        var summ = sum(sq, axies: .row)
        let epsilon: Double = 1e-10
        
//        print(m)
//        print(sq)
//        print(summ)
//        print("epsilon",epsilon)
        for r in 0..<summ.rows {
            if summ[row:r][0] < epsilon {
                summ[row:r] = [epsilon]
            }
        }
//        print(summ)
        
        let sq2 = myPow(summ, 0.5)
        var mm = m
        for r in 0..<mm.rows {
            let target = sq2[row:r][0]
            for (c, val) in m[row:r].enumerated() {
                mm[r, c] = val/target
            }
        }
//        print(sq2)
//        print("l2Normalize")
//        print(mm)
        return mm
    }
    
    func distanceMatrix(a:Matrix<Double>, b:Matrix<Double>) -> Double {
        let s = sum(myPow(a-b, 2), axies:.row)
        return sqrt(Double(s.grid[0]))
    }
    
    func diff() {
        print("--------diff--------")
        if let taniai1 = self.matrixDic["taniai1.png"],
        let taniai2 = self.matrixDic["taniai2.png"],
        let takemoto1 = self.matrixDic["takemoto1.png"],
        let takemoto2 = self.matrixDic["takemoto2.png"] {
            print("<<<same person>>>")
            let l1 = distanceMatrix(a:taniai1,b:taniai2)
            print("taniai1,taniai2:\(l1)")
            let l2 = distanceMatrix(a:takemoto1,b:takemoto2)
            print("takemoto1,takemoto2:\(l2)")
            
            print("<<<different person>>>")
            let l3 = distanceMatrix(a:taniai2,b:takemoto2)
            print("taniai2,takemoto2:\(l3)")
            let l4 = distanceMatrix(a:taniai1,b:takemoto1)
            print("taniai1,takemoto1:\(l4)")
            let l5 = distanceMatrix(a:taniai1,b:takemoto2)
            print("taniai1,takemoto2:\(l5)")
            let l6 = distanceMatrix(a:taniai2,b:takemoto1)
            print("taniai2,takemoto1:\(l6)")
        }
    }
    func diffNormalized() {
        print("--------diff diffNormalized--------")
        let taniai1 = Matrix([self.normlizedDic["taniai"]![row:0]])
        let taniai2 = Matrix([self.normlizedDic["taniai"]![row:1]])
        let takemoto1 = Matrix([self.normlizedDic["takemoto"]![row:0]])
        let takemoto2 = Matrix([self.normlizedDic["takemoto"]![row:1]])
        
        print("taniai dic ",self.normlizedDic["taniai"]!)
        print("taniai row0",[self.normlizedDic["taniai"]![row:0]])
//        print("taniai2",taniai2)
        print("**************")
//        print("takemoto dic ",self.normlizedDic["takemoto"]!)
//        print("takemoto row0",[self.normlizedDic["takemoto"]![row:0]])
        
        print("<<<same person>>>")
        let l1 = distanceMatrix(a:taniai1,b:taniai2)
        print("taniai1,taniai2:\(l1)")
        let l2 = distanceMatrix(a:takemoto1,b:takemoto2)
        print("takemoto1,takemoto2:\(l2)")
        
        print("<<<different person>>>")
        let l3 = distanceMatrix(a:taniai2,b:takemoto2)
        print("taniai2,takemoto2:\(l3)")
        let l4 = distanceMatrix(a:taniai1,b:takemoto1)
        print("taniai1,takemoto1:\(l4)")
        let l5 = distanceMatrix(a:taniai1,b:takemoto2)
        print("taniai1,takemoto2:\(l5)")
        let l6 = distanceMatrix(a:taniai2,b:takemoto1)
        print("taniai2,takemoto1:\(l6)")
        
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
                return $0.replacingOccurrences(of: " ", with: "").replacingOccurrences(of: "\n", with: "").components(separatedBy: ",").map{ Double($0)! }
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
    
    // MARK: - Prewhiten
    
    func mean(_ array: MultiArray<Double>) -> Double {
        var sum: Double = 0.0
        for i in 0..<array.count {
            sum += array[i]
        }
        //    print("sum",sum)
        //    print("mean", sum/Double(array.count))
        return sum/Double(array.count)
    }
    
    func std(_ array: MultiArray<Double>) -> Double {
        return sqrt(vars(array))
    }
    
    func stdAdj(_ array: MultiArray<Double>, std: Double) -> Double {
        let comp: Double = 1.0/sqrt(Double(array.count))
        return comp > std ? comp : std
    }
    
    // 不偏分散（標本分散）
    func vars(_ array: MultiArray<Double>) -> Double {
        return sumOfSquares(array) / Double(array.count)
    }
    
    // 平方和
    func sumOfSquares(_ array: MultiArray<Double>) -> Double {
        let mu: Double = mean(array)
        var ss: Double = 0.0
        for i in 0..<array.count {
            let deviation: Double = array[i] - mu
            ss += pow(deviation, 2.0)
        }
        return ss
    }
    
    func prewhiten(_ array: MultiArray<Double>) -> MultiArray<Double> {
        
        var marray = array
        let avg = mean(array)
        
        let s = std(array)
        
        let adj = stdAdj(array, std: s)
        
        for i in 0..<array.count {
            marray[i] = (array[i] - avg) * 1.0/adj
        }
        return marray
    }

}

