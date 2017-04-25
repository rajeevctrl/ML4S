package com.ml4s.util

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.StringWriter
import java.io.PrintWriter
import scala.reflect.ClassTag
import com.fasterxml.jackson.core.JsonParseException
import com.fasterxml.jackson.databind.JsonMappingException
import java.io.IOException
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import com.fasterxml.jackson.databind.ObjectWriter
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.core.JsonGenerationException

object CommonUtil {  
  private val commonUtil:CommonUtil=new CommonUtil()
  private val logger:Logger=commonUtil.logger
  
  /**
   * Generates an array of length {@code n} of random double values.
   * 
   * @param start Starting of range.
   * @param end Ending of range.
   * 
   * @return An array of generated double random numbers. 
   */
  def randomNumbersInRange(n:Int, start:Double=0.0, end:Double=0.001):Array[Double]={
    val arr=Array.fill[Double](n)(0)
    return arr
  }
  
  def formatException(e:Exception):String={
    val writer:StringWriter=new StringWriter()
    val printWriter:PrintWriter=new PrintWriter(writer)
    e.printStackTrace(printWriter)
    printWriter.flush()
    val stackTrace=writer.toString()
    printWriter.close()
    writer.close()
    /*val exceptionMsg=CommonUtil.formatException(e.getStackTrace)
    return exceptionMsg*/
    return  stackTrace
  }
  
  
  
  def convertJsonToObject[T](json:String, classType:Class[T])(implicit tag:ClassTag[T]):T={
    val mapper:ObjectMapper=new ObjectMapper() with ScalaObjectMapper
    mapper.registerModule(DefaultScalaModule)
    
    var result:T= null.asInstanceOf[T]
    if(json==null) return null.asInstanceOf[T]
    try {
			result = mapper.readValue(json.getBytes(), classType);
		} catch{
		  case e:JsonParseException => logger.error(formatException(e))
		  case e:JsonMappingException => logger.error(formatException(e))
		  case e:IOException => logger.error(formatException(e))
		}    
    return result
  }
  
  /**
   * This function converts an object to JSON string.
   */
  def convertObjectToJson(obj:Object):String={
    val mapper:ObjectMapper=new ObjectMapper() with ScalaObjectMapper
    mapper.registerModule(DefaultScalaModule)
    //val jsonWriter:ObjectWriter=mapper.writer().withDefaultPrettyPrinter()
    val jsonWriter:ObjectWriter=mapper.writer()
    var json:String=null
    try{
      json = jsonWriter.writeValueAsString(obj)
    }catch{
      case e:JsonGenerationException => logger.error(formatException(e))
      case e:JsonMappingException => logger.error(formatException(e))
      case e:IOException => logger.error(formatException(e))
    }
    return json
  }
}

class CommonUtil{
  private val logger:Logger=LoggerFactory.getLogger(classOf[CommonUtil])
  
  
}