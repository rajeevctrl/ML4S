package com.ml4s.np

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits.RichINDArray
import org.nd4s.Implicits._
//import com.ml4s.np.temp.NPImplicits._
import scala.collection.GenIterable

//import com.ml4s.np.temp.NP

object NPImplicits {
  
  
//  implicit class IntRangeFrom(val x:Int) extends AnyVal{
//    /** Generates a range of numbers from this.x to y(non-inclusive). */
//    def -> (y:Int):IndexRange= return new IndexRangeInt(Range(x,y).toArray)
//  }
  
  implicit object np extends NPTrait
  
  /**
   * This class will be used to provide extra methods for all numeric types.
   */
  implicit class NPNumber[T](x:T)(implicit n:Numeric[T]){
    /**
     * Element-wise add this.x with given array.
     */
    def +(arr:INDArray):INDArray = return arr.add(n.toDouble(x))
    /**
     * Element-wise subtract this.x with given array.
     */
    def -(arr:INDArray):INDArray = {
      val scalar = NP.scalars(this.x,shape=arr.shape())
      val diff = scalar.sub(arr)
      return diff
    }
    /**
     * Element-wise multiply this.x with given array.
     */
    def *(arr:INDArray):INDArray = return arr.mul(n.toDouble(x))
    /**
     * Element-wise division of this.x with given array.
     */
    def /(arr:INDArray):INDArray={
      val scalar = NP.scalars(this.x, shape=arr.shape())
      val div = scalar.div(arr)
      return div
    }    
    
    def to(start:Int):Array[Int]={
      val end = n.toInt(x)
      val range=Range(start,end,1).toArray
      return range
    }
    
  }
   
  //case object -> extends IndexRange
  /**
   * This class is used to provide extra methods to objects of type INDArray.
   */
//  implicit class NPArray [A <: INDArray](val arr: A) extends RichINDArray(arr){
//    //override val underlying = arr
//
//    
//    
//    def apply(dims:IndexRange*)={
//      println("apply called")
//      val shape = this.arr.shape()
//      val seqIndex=Range(0,dims.size).toSeq
//      val enumerated = seqIndex.zip(dims)
//      for((index,dim) <- enumerated){
//         dim match{
//           case dim:IndexRangeInt => {
//             println(s"${index}  dim of type IndexRangeInt len=${dim.arr.size} indicies={${dim.arr.toList}}")
//           }
//           case -> => {
//             val len=this.arr.size(index)
//             println(s"${index}  dim of type -> len=${len} indicies={${Range(0,len).toList}}")
//           }
//         }
//      }
//    }
//    
//    def subArray(indicies:Array[Int]*)={
//      //this.arr.subArray
//    }
//    
//    /**
//     * Element-wise add this.arr to num.
//     */
//    def +[T](num:T)(implicit n:Numeric[T]):INDArray={
//      val x=n.toDouble(num)
//      return this.arr.add(x)
//    }
//    def +[T <:INDArray](that:A):INDArray={
//      return this.arr.add(that.broadcast(this.arr.shape: _*))
//    }
//    
//    /**
//     * Element-wise subtract this.arr to num.
//     */
//    def -[T](num:T)(implicit n:Numeric[T]):INDArray={
//      val x=n.toDouble(num)
//      return this.arr.sub(x)
//    }
//    def -(that:INDArray):INDArray={
//      return this.arr.sub(that.broadcast(this.arr.shape: _*))
//    }
////    def unary_- :INDArray = {
////      return this.arr.mul(-1)
////    }
//    /**
//     * Element-wise multiply this.arr to num.
//     */
//    def *[T](num:T)(implicit n:Numeric[T]):INDArray={
//      val x=n.toDouble(num)
//      return this.arr.mul(x)
//    }
//    def *(that:INDArray):INDArray={
//      return this.arr.mul(that.broadcast(this.arr.shape: _*))
//    }
//    /**
//     * Element-wise divide this.arr to num.
//     */
//    def /[T](num:T)(implicit n:Numeric[T]):INDArray={
//      val x=n.toDouble(num)
//      return this.arr.div(x)
//    }
//    def /(that:INDArray):INDArray={
//      return this.arr.div(that.broadcast(this.arr.shape:_*))
//    }
//    
//    
//  }
  
//  implicit object -> extends IndexRange {
//    
//  }
//  sealed trait IndexRange
//  class IndexRangeInt(val arr:Array[Int]) extends IndexRange
}





