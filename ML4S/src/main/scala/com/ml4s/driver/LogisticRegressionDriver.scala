package com.ml4s.driver

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import com.ml4s.regression.LogisticRegression
import com.ml4s.hypothesis.Hypothesis
import com.ml4s.hypothesis.HypothesisLogistic
import com.ml4s.costfunctions.CostFunction
import com.ml4s.costfunctions.CostFunctionSquaredDiff
import org.nd4j.linalg.api.ndarray.INDArray
import com.ml4s.pojo.model.ModelRegression
import org.nd4j.linalg.factory.Nd4j
import java.util.regex.Pattern
import com.ml4s.pojo.model.ModelLogisticRegression
import com.ml4s.util.CommonUtil
import java.io.File
import java.nio.file.Path
import java.nio.file.Files
import java.nio.file.Paths

class LogisticRegressionDriver {
  private val logger:Logger=LoggerFactory.getLogger(classOf[LinearRegressionDriver])
  
  private val logisticRegression:LogisticRegression=new LogisticRegression();
  private val modelName="LogisticRegressionModel.model";
  private var hypothesis:Hypothesis=new HypothesisLogistic();
  private var costFunction:CostFunction=new CostFunctionSquaredDiff();
  
  private var X:INDArray = _
  private var Y:INDArray = _
  private var theta:INDArray = _
  private var trainingError:Double = _
  private var alpha:Double = _
  private var iter:Int= _
  
  def getTheta():INDArray=return this.theta
  def setTheta(theta:INDArray)= this.theta=theta
  def getHypothesis():Hypothesis=this.hypothesis
  def getTrainingError():Double=this.trainingError
  
  def this(X:INDArray, Y:INDArray, alpha:Double, iter:Int)={
    this();
    this.X=X;
    this.Y=Y;
    this.alpha=alpha;
    this.iter=iter;
  }
  
  /**
   * This function predicts output values for given feature vectors of matrix {@code X}
   * 
   * @param X: INDArray of new data points for which output values are to be predicted.
   */
  def predict(X:INDArray):INDArray={
    val predictedValues=this.logisticRegression.predict(X, this.theta,this.hypothesis);
    return predictedValues;
  }
  
  /**
   * This function fits a Linear regression model with {@code HypothesisLinear} as hypothesis function
   * and {@code CostFunctionSquaredDiff} as cost function.
   */
  def fitModel()={
    this.logger.info("Starting Linear Regression learning.")  
    this.logisticRegression.fit(X, Y, alpha, iter, this.hypothesis, this.costFunction)
    //val theta=this.linearRegression.getThetaINDArray();
    this.theta=this.logisticRegression.getThetaINDArray();;
    this.trainingError=this.logisticRegression.getTrainingError()
    this.logger.info(s"Final value of Training Error:${this.trainingError}, theta:${this.theta}")
    this.logger.info("Finishing Logistic Regression learning.")
    //return this.model
  }
  
  /**
   * This function saves linear regression model to given {@code modelFilePath}
   * 
   * @param modelFilePath: Complete path (including file name) to which model will be saved.
   * 
   */
  def saveModel(modelFilePath:String)={
    var filePath=modelFilePath.replaceAll(Pattern.quote("\\"), "/")      
    //Handle spaces in path.
    filePath=filePath.replaceAll(" ", "\\ ")
    //Find path excluding filename.
    val homeFolder=filePath.substring(0, filePath.lastIndexOf("/"))
    //Create model class to be serialized to JSON.
    val thetaArr=(for(i <- Range(0,theta.size(0))) yield {
      theta.getDouble(i)
    }).toArray
    val logisticModel=ModelLogisticRegression(thetaArr,this.trainingError);
    //Serialize model class to JSON.
    val json=CommonUtil.convertObjectToJson(logisticModel)
    val file=new  File(homeFolder)
    //Create dir structure upto home folder if not already created.
    if(!file.exists()){
      file.mkdirs();
    }
    //Save JSON string of model to given file path.
    var outputPath:Path=null;
    try{
      outputPath=Files.write(Paths.get(filePath),json.getBytes())
      this.logger.info(s"Model saved successfully to file ${modelFilePath}")
    }catch{
      case e:Exception => {
        this.logger.error(s"Error saving model to file ${modelFilePath}")
        this.logger.error(CommonUtil.formatException(e))
      }
    }
  }
  
  def loadModel(modelFilePath:String)={
    var filePath=modelFilePath.replaceAll(Pattern.quote("\\"), "/")      
    //Handle spaces in path.
    filePath=filePath.replaceAll(" ", "\\ ")
    val json:String=new String(Files.readAllBytes(Paths.get(filePath)))
    val logisticModel=CommonUtil.convertJsonToObject(json, classOf[ModelLogisticRegression])
    //update values of class variables from model parameters.
    this.explodeModel(logisticModel)
    this.logger.info(s"Loaded model from file ${modelFilePath}")
  }
  
  /**
   * This function extracts parameters from model and saves them to variables of this class.
   */
  private def explodeModel(model:ModelRegression)={    
    this.theta=Nd4j.create(model.theta,Array(model.theta.length,1))
    this.trainingError=model.trainingError
    //this.costFunction=model.costFunction
    //this.hypothesis=model.hypothesis
  }
}