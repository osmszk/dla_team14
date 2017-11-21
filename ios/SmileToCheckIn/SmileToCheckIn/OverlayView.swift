//
//  OverlayView.swift
//  SmileToCheckIn
//
//  Created by 鈴木治 on 2017/11/21.
//  Copyright © 2017年 Osamu Suzuki. All rights reserved.
//

import UIKit

class OverlayView: UIView {

    var boxes:[CGRect]?
    
    override func draw(_ rect: CGRect) {
        let context = UIGraphicsGetCurrentContext()
        
        boxes?.forEach { (box) in
            context?.setStrokeColor(UIColor.red.cgColor)
            context?.setLineWidth(5)
            context?.stroke(box)
        }
    }
}
