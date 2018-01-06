//
//  SmileToCheckInTests.swift
//  SmileToCheckInTests
//
//  Created by 鈴木治 on 2017/12/29.
//  Copyright © 2017年 Osamu Suzuki. All rights reserved.
//

import XCTest
import CoreML

class SmileToCheckInTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        let a = Matrix([[1,2],[3,4],[5,6],[7,8]])
        print(a.description)
        print(a.grid)
        print(sum(a.grid))
        print(sum(a))
        print(a.columns * a.rows)
        print(sum(a.grid)/Double(a.columns * a.rows))
        print(mean(matrix: a))
    }
    
    func testDistance() {
        
        let a = Matrix(Array(repeating: [1,2,3], count: 1))
        let b = Matrix(Array(repeating: [10,20,30], count: 1))
        
        let d = sum(myPow(a-b, 2), axies:.row)
        print(d.grid[0])
        let dis = sqrt(Double(d.grid[0]))
        print(dis)
        
        print(distanceMatrix(a: a, b: b))
        
    }
    
    func distanceMatrix(a:Matrix<Double>, b:Matrix<Double>) -> Double {
        let s = sum(myPow(a-b, 2), axies:.row)
        return sqrt(Double(s.grid[0]))
    }
    
    func testL2NormalizeMarix() {
        let a: [Double] = [0,1,2,3,6,2]
        let b: [Double] = [7,8,4,5,3,3]
        let c: [Double] = [1,2,3,5,6,9]
        let concat: [[Double]] = [a,b,c]
        //        let row = 3
        //        let column = 4
        let m = Matrix<Double>(concat)
        print(m)
        let sq = myPow(m, 2)
        var summ = sum(sq, axies: .row)
        let epsilon: Double = 1e-10
        
        print(sq)
        print(summ)
        print("epsilon",epsilon)
        for r in 0..<summ.rows {
            if summ[row:r][0] < epsilon {
                summ[row:r] = [epsilon]
            }
        }
        print(summ)
        
        let sq2 = myPow(summ, 0.5)
        var mm = m
        for r in 0..<mm.rows {
            let target = sq2[row:r][0]
            for (c, val) in m[row:r].enumerated() {
                mm[r, c] = val/target
            }
        }
        print(sq2)
        print(mm)
    }
    
    
    
    func mean(matrix:Matrix<Double>) -> Double {
        return sum(matrix.grid)/Double(matrix.columns * matrix.rows)
    }
    
    
}
