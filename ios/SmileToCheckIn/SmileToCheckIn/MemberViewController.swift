//
//  MemberViewController.swift
//  SmileToCheckIn
//
//  Created by 鈴木治 on 2017/12/02.
//  Copyright © 2017年 Osamu Suzuki. All rights reserved.
//

import UIKit
import Metal
import MetalPerformanceShaders
import AVFoundation
import CoreMedia

class MemberViewController: UIViewController {

    @IBOutlet weak var videoPreview: UIView!
    @IBOutlet weak var timeLabel: UILabel!
    
    let maxBoundingBoxes = 10
    
    var videoCapture: VideoCapture!
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
//    var runner: Runner!
    
    var startupGroup = DispatchGroup()
    
    var boundingBoxes = [BoundingBox]()
    var colors: [UIColor] = []
    let fpsCounter = FPSCounter()
    
    override func viewDidLoad() {
        super.viewDidLoad()

        timeLabel.text = ""
        
        device = MTLCreateSystemDefaultDevice()
        if device == nil {
            print("Error: this device does not support Metal")
            return
        }
        
        commandQueue = device.makeCommandQueue()
        
        // Set up the bounding boxes.
        for _ in 0..<maxBoundingBoxes {
            boundingBoxes.append(BoundingBox())
        }
        // Make colors for the bounding boxes. There is one color for each class,
        // 20 classes in total.
        for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
            for g: CGFloat in [0.3, 0.7] {
                for b: CGFloat in [0.4, 0.8] {
                    let color = UIColor(red: r, green: g, blue: b, alpha: 1)
                    colors.append(color)
                }
            }
        }
        
        videoCapture = VideoCapture(device: device)
        videoCapture.delegate = self
        
        // Initialize the camera.
        startupGroup.enter()
        videoCapture.setUp(sessionPreset: .vga640x480) { success in
            // Add the video preview into the UI.
            if let previewLayer = self.videoCapture.previewLayer {
                self.videoPreview.layer.addSublayer(previewLayer)
                self.resizePreviewLayer()
            }
            self.startupGroup.leave()
        }
        
//        startupGroup.enter()
//        runner = Runner(commandQueue: commandQueue, inflightBuffers: MaxBuffersInFlight)
        
        
        startupGroup.notify(queue: .main) {
            // Add the bounding box layers to the UI, on top of the video preview.
            for box in self.boundingBoxes {
                box.addToLayer(self.videoPreview.layer)
            }
            
            // Once the NN is set up, we can start capturing live video.
            self.fpsCounter.start()
            self.videoCapture.start()
        }
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
        print(#function)
    }
    
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        resizePreviewLayer()
    }
    
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }
    
    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = videoPreview.bounds
    }
    

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destinationViewController.
        // Pass the selected object to the new view controller.
    }
    */
    
    func predict(texture: MTLTexture) {
        // Since we want to run in "realtime", every call to predict() results in
        // a UI update on the main thread. It would be a waste to make the neural
        // network do work and then immediately throw those results away, so the
        // network should not be called more often than the UI thread can handle.
        // It is up to VideoCapture to throttle how often the neural network runs.
        
//        runner.predict(network: network, texture: texture, queue: .main) { result in
//            self.show(predictions: result.predictions)
//
//            if let texture = result.debugTexture {
//                self.debugImageView.image = UIImage.image(texture: texture)
//            }
//
//            self.fpsCounter.frameCompleted()
//            self.timeLabel.text = String(format: "%.1f FPS (latency: %.5f sec)", self.fpsCounter.fps, result.latency)
//        }
        
        self.fpsCounter.frameCompleted()
        self.timeLabel.text = String(format: "%.1f FPS (latency: xxx sec)", self.fpsCounter.fps/*, result.latency*/)
    }

}

extension MemberViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoTexture texture: MTLTexture?, timestamp: CMTime) {
        // Call the predict() method, which encodes the neural net's GPU commands,
        // on our own thread. Since NeuralNetwork.predict() can block, so can our
        // thread. That is OK, since any new frames will be automatically dropped
        // while the serial dispatch queue is blocked.
        if let texture = texture {
            predict(texture: texture)
        }
    }
    
    func videoCapture(_ capture: VideoCapture, didCapturePhotoTexture texture: MTLTexture?, previewImage: UIImage?) {
        // not implemented
    }
}
