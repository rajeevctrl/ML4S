package com.ml4s.np.temp

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import scala.collection.Iterable
import org.nd4j.linalg.factory.Nd4j
import scala.Range
import scala.collection.Seq
import org.nd4s.SliceableNDArray
import org.nd4s.OperatableNDArray
import org.nd4s.CollectionLikeNDArray
import org.nd4j.linalg.inverse.InvertMatrix
import com.ml4s.np.temp.NPImplicits._
import org.nd4s.Implicits._


final object NP {
  
  ////////////////////// Implicit classes ////////////////////////
  
  /**
   * Object specific to random numbers.
   */
  object random {
    /**
     * Creates and array with gaussian generated data of given shape.
     */
    def randn(shape: Array[Int]): INDArray = {
      return Nd4j.randn(shape)
    }
  }
  
  /////////////////////////// NP Specific methods ///////////////////////////////////
  
  /**
   * Takes an array like structure i.e. multidimensional array of Sequences and converts them to INDArray.
   * Maximum allowed dimensions are 3D.
   */
  def array(arrayLike:Any,shape:Array[Int]=Array[Int](),order:Char='r'):INDArray={
    val indArray:INDArray = this.createNDArray(arrayLike, shape)
    return indArray
  }
  
  /**
   * Creates array of zeros for given shape.
   */
  def zeros(shape:Array[Int]):INDArray={
    return Nd4j.zeros(shape: _*)
  }
  
  /**
   * Creates a square identity matrix of shape n x n.
   */
  def identity(n:Int):INDArray={
    val identity = Nd4j.eye(n)
    return identity
  }
  
  /**
   * Creates array of ones for given shape.
   */
  def ones(shape:Array[Int]):INDArray={
    return Nd4j.ones(shape: _*)
  }
  
  /**
   * Converts {@code arr} to column vector.
   */
  def toColVector[T](arr:Array[T])(implicit n:Numeric[T]):INDArray={
    val doubleArr = arr.map(x => n.toDouble(x))
    val shape = Array(doubleArr.size,1)
    val result = array(doubleArr,shape)
    return result
  }
  /**
   * Converts {@code arr} to column vector.
   */
  def toColVector(arr:INDArray):INDArray={
    val result=if(arr.shape()(0)>1 && arr.shape()(1)==1){
      arr
    }else if(arr.shape()(0)==1 && arr.shape()(1)>1){
      reshape(arr,newshape=Array(arr.shape()(1),arr.shape()(0)))
    }else{
      throw new Exception(s"Input array should be a vector. However it has shape ${arr.shape().toList}")
    }
    return result
  }
  
  /**
   * Computes inverse of matrix.
   */
  def inv(arr:INDArray,inplace:Boolean=false):INDArray={
    assert(arr.shape()(0) == arr.shape()(1),"Input matrix 'arr' should be a square matrix.")
    return InvertMatrix.invert(arr, inplace);
  }  
  
  /**
   * Computes Moore-Penrose pseudo-inverse.
   */
  def pinv(arr:INDArray,alpha:Double=0.00001):INDArray={
    //val xTx = this.matmul(arr.transpose(), arr)
    val xTx = arr
    //val pinverse = this.matmul(this.inv(xTx + (alpha * this.identity(xTx.shape()(0)))), arr.transpose())
    val pinverse = this.inv(xTx + (alpha * this.identity(xTx.shape()(0))))
    return pinverse
  }
  
  
  /**
   * Creates array of given {@code scalar} for given {@code shape}.
   */
  def scalars[T](scalar:T,shape:Array[Int])(implicit n:Numeric[T]):INDArray={
    val x = n.toDouble(scalar)
    val ones = Nd4j.ones(shape: _*)
    val result = multiply(ones, x)
    return result
  }
  
  /**
   * Element wise log. If {@code base} is not provided then default base=e is used.
   */
  def log[T](arr:INDArray,base:T)(implicit n:Numeric[T]):INDArray={
    var base_ = n.toDouble(base)
    val result = if(base_ == 0.0){
      Transforms.log(arr)
    }else{
      Transforms.log(arr,base_)
    }
    return result
  }
  
  /**
   * Element wise sine transform.
   */
  def sin(arr:INDArray):INDArray={
    return Transforms.sin(arr)
  }
  
  /**
   * Element wise cosine transform.
   */
  def cos(arr:INDArray):INDArray={
    return Transforms.cos(arr)
  }
  
  /**
   * Element wise tan transform.
   */
  def tan(arr:INDArray):INDArray={
    val sine = Transforms.sin(arr)
    val cosine = Transforms.cos(arr)
    val tan = sine.div(cosine)
    return tan
  }
  
  
  /**
   * Element wise exponentiation.
   */
  def exp(arr:INDArray):INDArray={
    return Transforms.exp(arr)
  }
  
  /**
   * Element wise absolute value.
   */
  def abs(arr:INDArray):INDArray={
    return Transforms.abs(arr)
  }
  
  /**
   * Max value {@code arr} for given dimensions. If dimensions are not given then max value of complete array is returned.
   */
  def max(arr:INDArray, dims:Iterable[Int]=Seq[Int]()):INDArray={
    val dimensions = if(dims.size==0) Range(0,dims.size).toArray else dims.toArray
    val maxArr=arr.max(dimensions: _*)
    return maxArr
  }
  
  /**
   * Min value {@code arr} for given dimensions. If dimensions are not given then min value of complete array is returned.
   */
  def min(arr:INDArray, dims:Iterable[Int]=Seq.empty[Int]):INDArray={
    val dimensions = if (dims.size == 0) Range(0,dims.size).toArray else dims.toArray
    val minArr = arr.min(dimensions: _*)
    return minArr
  }
  
  /**
   * Appends given {@code arrays} horizontally. 
   */
  def hstack(arrays:INDArray*):INDArray={
    return Nd4j.hstack(arrays: _*)
  }
  /**
   * Appends given {@code arrays} vertically. 
   */
  def vstack(arrays:INDArray*):INDArray={
    return Nd4j.vstack(arrays: _*)
  }
  
  /**
   * Element wise multiplication.
   */
  def multiply(arr1:INDArray,arr2:INDArray):INDArray={
    return arr1.mul(arr2)
  }
  def multiply[T](arr1:INDArray,num:T)(implicit n:Numeric[T]):INDArray={
    val num1 = n.toDouble(num)
    return arr1.mul(num1)
  }
  def multiply[T](num:T,arr1:INDArray)(implicit n:Numeric[T]):INDArray={
    val scalar = scalars(num, shape=arr1.shape())
    return multiply(scalar,arr1)
  }
  
  /**
   * Element wise addition of two arrays.
   */  
  def add(arr1:INDArray,arr2:INDArray):INDArray={
    return arr1.add(arr2)
  }
  def add[T](arr:INDArray,num:T)(implicit n:Numeric[T]):INDArray={
    val num1:Double = n.toDouble(num)
    return arr.add(num1)
  }
  def add[T](num:T,arr:INDArray)(implicit n:Numeric[T]):INDArray={
    val scalar = scalars(num,shape=arr.shape)
    return add(scalar,arr)
  }
  
  /**
   * Element wise subtraction of elements of two arrays.
   */
  def subtract(arr1:INDArray,arr2:INDArray):INDArray={
    return arr1.sub(arr2)
  }
  def subtract[T](arr:INDArray,num:T)(implicit n:Numeric[T]):INDArray={
    val num1 = n.toDouble(num)
    return arr.sub(num1)
  }
  def subtract[T](num:T,arr:INDArray)(implicit n:Numeric[T]):INDArray={
    val scalar = scalars(num,arr.shape)
    return subtract(scalar,arr)
  }
  
  /**
   * Element wise division of elements of {@code arr1} with {@code arr2}.
   */
  def divide(arr1:INDArray,arr2:INDArray):INDArray={
    return arr1.div(arr2)
  }
  def divide[T](arr:INDArray,num:T)(implicit n:Numeric[T]):INDArray={
    val num1=n.toDouble(num)
    return arr.div(num1)
  }
  def divide[T](num:T,arr:INDArray)(implicit n:Numeric[T]):INDArray={
    val scalar = scalars(num,shape=arr.shape())
    return divide(scalar,arr)
  }
  
  /**
   * Matrix multiplication. Columns of {@code arr1} should match with rows of {@code arr2}.
   */
  def matmul(arr1:INDArray,arr2:INDArray):INDArray={
    return arr1.mmul(arr2)
  }
  
  /**
   * Reshapes array.
   */
  def reshape(arr:INDArray,newshape:Array[Int]):INDArray={
    return arr.reshape(newshape: _*)
  }
  
  
  
  /**
   * This function computes ordeding of INDArray. Following is the mapping for user input and corresponding internal implementation:
   * <ol>
   * <li> c = column ordering = Internal implementation is 'F' meaning FORTRAN based ordering.</li>
   * <li> r = row ordering = Internal implementation is 'C' meaning C based ordering. </li>
   * </ol>
   */
  private def ordering(ord:Char):Char={
    val order = if(ord == 'c'){ 
      'F' 
    }else{
      'C'
    }
    return order
  }
  
  /**
   * This functio takes array likes of upto 3D and converts them to 1D and also finds their shape.
   * 
   * @return (1D array, identified shape)
   */
  private def createNDArray(arrayLike:Any,inputShape:Array[Int]):INDArray={
    val ndArray = arrayLike match{
      //Int arrays upto 3D.
      case arr:Array[Int] => {
        val a = arr.map(_.toFloat)
        val s = if(inputShape.size==0) Array(arr.size,1) else inputShape
        Nd4j.create(a,s)
      }case arr:Array[Array[Int]] => {
        val a = arr.flatMap(d1 => d1).map(_.toFloat)
        val s = if(inputShape.size==0) Array(arr.size,arr(0).size) else inputShape
        Nd4j.create(a,s)
      }case arr:Array[Array[Array[Int]]] => {
        val a = arr.flatMap(d1 => d1).flatMap(d2 => d2).map(_.toFloat)
        val s = if(inputShape.size==0) Array(arr.size,arr(0).size,arr(0)(0).size) else inputShape
        Nd4j.create(a,s)
      }
      //Double arrays upto 3D
      case arr:Array[Double] => {
        val a = arr
        val s = if(inputShape.size==0) Array(arr.size,1) else inputShape
        Nd4j.create(a,s)
      }case arr:Array[Array[Double]] => {
        val a = arr.flatMap(d1 => d1)
        val s = if(inputShape.size==0) Array(arr.size,arr(0).size) else inputShape
        Nd4j.create(a,s)
      }case arr:Array[Array[Array[Double]]] => {
        val a = arr.flatMap(d1 => d1).flatMap(d2 => d2)
        val s = if(inputShape.size==0) Array(arr.size,arr(0).size,arr(0)(0).size) else inputShape
        Nd4j.create(a,s)
      }
      //Default case
      case _ => {
        println("Input is not valid array.")
        throw new Exception("Input is not valid array.")
      }
//      //Seq[Int] upto 3D
//      case Arr1DInt(arr) => {
//        val a = arr.map(_.toDouble).toArray
//        val s = Array(arr.size,1)
//        (a,s)
//      }case Arr2DInt(arr) => {
//        val a = arr.flatMap(d1 => d1).map(_.toDouble).toArray
//        val s = Array(arr.size,arr(0).size)
//        (a,s)
//      }case Arr3DInt(arr) => {
//        val a = arr.flatMap(d1 => d1).flatMap(d2 => d2).map(_.toDouble).toArray
//        val s = Array(arr.size,arr(0).size,arr(0)(0).size)
//        (a,s)
//      }
//      //Seq[Double] upto 3D
//      case Arr1DDouble(arr) => {
//        val a = arr.map(_.toDouble).toArray
//        val s = Array(arr.size,1)
//        (a,s)
//      }case Arr2DDouble(arr) => {
//        val a = arr.flatMap(d1 => d1).map(_.toDouble).toArray
//        val s = Array(arr.size,arr(0).size)
//        (a,s)
//      }case Arr3DDouble(arr) => {
//        val a = arr.flatMap(d1 => d1).flatMap(d2 => d2).map(_.toDouble).toArray
//        val s = Array(arr.size,arr(0).size,arr(0)(0).size)
//        (a,s)
//      }
      
      
    }
    
    return ndArray
  }
  
  
}