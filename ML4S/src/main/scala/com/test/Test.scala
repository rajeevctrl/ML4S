package com.test

//import com.ml4s.np.{ NP => np }
import com.ml4s.np.NPImplicits._
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms
import com.ml4s.sklearn.activations.Relu
import com.ml4s.sklearn.activations.Softmax

object Test {  
  
  def main(args:Array[String]):Unit={
     val arr = np.array(Array(1,2,3,4,5,6), Array(2,3))
     var sum = 1 + arr * 2
     print(sum)
  }
  
  
}