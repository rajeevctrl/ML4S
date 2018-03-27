package com.ml4s.np

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4s.SliceableNDArray
import org.nd4s.OperatableNDArray
import org.nd4s.CollectionLikeNDArray


object _NPImplicits{
  /**
   * This class will be used to provide extra methods for all numeric types.
   */
  implicit class NPNumber[T](x:T)(implicit n:Numeric[T]){
    /**
     * Element-wise add this.x with given array.
     */
    def +(arr:INDArray):INDArray = {
      return arr.add(n.toDouble(x))
    }
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
    def *(arr:INDArray):INDArray={
      return arr.mul(n.toDouble(x))
    }
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
    
    def :: (y:Int)={
      Range(this.n.toInt(x),y)
    }
  }
  
  /**
   * This class is used to provide extra methods to objects of type INDArray.
   */
  implicit class NPArray [A <: INDArray](val arr: A) extends SliceableNDArray[A] with OperatableNDArray[A] with CollectionLikeNDArray[A]{
    override val underlying = arr
    
    
    /**
     * Element-wise add this.arr to num.
     */
    def +[T](num:T)(implicit n:Numeric[T]):INDArray={
      val x=n.toDouble(num)
      return this.arr.add(x)
    }
    def +(that:INDArray):INDArray={
      return this.arr.add(that.broadcast(this.arr.shape: _*))
    }
    
    /**
     * Element-wise subtract this.arr to num.
     */
    def -[T](num:T)(implicit n:Numeric[T]):INDArray={
      val x=n.toDouble(num)
      return this.arr.sub(x)
    }
    def -(that:INDArray):INDArray={
      return this.arr.sub(that.broadcast(this.arr.shape: _*))
    }
//    def unary_- :INDArray = {
//      return this.arr.mul(-1)
//    }
    /**
     * Element-wise multiply this.arr to num.
     */
    def *[T](num:T)(implicit n:Numeric[T]):INDArray={
      val x=n.toDouble(num)
      return this.arr.mul(x)
    }
    def *(that:INDArray):INDArray={
      return this.arr.mul(that.broadcast(this.arr.shape: _*))
    }
    /**
     * Element-wise divide this.arr to num.
     */
    def /[T](num:T)(implicit n:Numeric[T]):INDArray={
      val x=n.toDouble(num)
      return this.arr.div(x)
    }
    def /(that:INDArray):INDArray={
      return this.arr.div(that.broadcast(this.arr.shape:_*))
    }
  }
  
//  implicit object -> extends IndexRange {
//    
//  }
  
}






