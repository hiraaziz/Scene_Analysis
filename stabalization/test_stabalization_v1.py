#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from stabalization.stabalization_v1 import MATLABStyleStabilizer

def test_stabalization():
    """Test MATLAB-style video stabilization on test11.mp4"""
    print("Testing MATLAB-Style Video Stabilization")
    print("=" * 50)
    
    # Initialize stabilizer
    stabilizer = MATLABStyleStabilizer(pt_thresh=0.1)
    
    # Test video paths
    input_video = "test11.mp4"
    output_video = "meshflow_stabilized.mp4"
    analysis_plot = "meshflow_matlab_analysis_test.png"
    
    print(f"Input: {input_video}")
    print(f"Output: {output_video}")
    print(f"Analysis plot: {analysis_plot}")
    print()
    
    # Perform stabilization
    print("Starting stabilization...")
    success = stabilizer.stabilize_video(input_video, output_video)
    
    if success:
        print("\nStabilization completed successfully!")
        
        # Perform analysis
        print("Starting stability analysis...")
        results = stabilizer.analyze_stability(input_video, output_video, analysis_plot)
        
        if results:
            print("\nRESULTS SUMMARY:")
            print(f"   Motion reduction: {results.get('motion_reduction', 0):.2f}%")
            print(f"   Stability improvement: {results.get('stability_improvement', 0):.2f}%")
            print(f"   Smoothness improvement: {results.get('smoothness_improvement', 0):.2f}%")
            print(f"   Overall score: {results.get('overall_score', 0):.2f}%")
            print(f"   Quality: {results.get('quality', 'Unknown')}")
            
            if results.get('is_improved', False):
                print("\nSUCCESS: Video stabilization achieved improvement!")
            else:
                print("\nLIMITED: Minimal improvement detected")
        
    else:
        print("Stabilization failed!")
        return False
    
    return True

if __name__ == "__main__":
    test_stabalization() 