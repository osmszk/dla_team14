//
//  SmileToCheckInTests.swift
//  SmileToCheckInTests
//
//  Created by 鈴木治 on 2017/12/29.
//  Copyright © 2017年 Osamu Suzuki. All rights reserved.
//

import XCTest

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
    
    func mean(matrix:Matrix<Double>) -> Double {
        return sum(matrix.grid)/Double(matrix.columns * matrix.rows)
    }
    
    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }
    
}
