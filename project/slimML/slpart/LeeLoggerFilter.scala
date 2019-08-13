package slpart

import org.apache.log4j._
import java.nio.file.{Paths, Files}

/**
  * logger filter, which will filter the log of Spark(org, breeze, akka) to file.
  */
object LeeLoggerFilter {
  private val pattern = "%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n"

  /**
    * redirect log to `filePath`
    *
    * @param filePath log file path
    * @param level logger level, the default is Level.INFO
    * @return a new file appender
    */
  private def fileAppender(filePath: String, level: Level = Level.INFO): FileAppender = {
    val fileAppender = new FileAppender
    fileAppender.setName("FileLogger")
    fileAppender.setFile(filePath)
    fileAppender.setLayout(new PatternLayout(pattern))
    fileAppender.setThreshold(level)
    fileAppender.setAppend(true)
    fileAppender.activateOptions()

    fileAppender
  }

  /**
    * redirect log to console or stdout
    *
    * @param level logger level, the default is Level.INFO
    * @return a new console appender
    */
  private def consoleAppender(level: Level = Level.INFO): ConsoleAppender = {
    val console = new ConsoleAppender
    console.setLayout(new PatternLayout(pattern))
    console.setThreshold(level)
    console.activateOptions()
    console.setTarget("System.out")

    console
  }

  /**
    * find the logger of `className` and add a new appender to it.
    *
    * @param className class which user defined
    * @param appender appender, eg. return of `fileAppender` or `consoleAppender`
    */
  private def classLogToAppender(className: String, appender: Appender): Unit = {
    Logger.getLogger(className).addAppender(appender)
  }
  private val defaultPath = Paths.get(System.getProperty("user.dir"), "run.log").toString


  def redirectSparkInfoLogs(classNames: Option[Array[String]] = None,logPath: String = defaultPath): Unit = {

    def getLogFile: String = {
      val logFile = System.getProperty("bigdl.utils.LoggerFilter.logFile", logPath)

      // If the file doesn't exist, create a new one. If it's a directory, throw an error.
      val logFilePath = Paths.get(logFile)
      if (!Files.exists(logFilePath)) {
        Files.createFile(logFilePath)
      } else if (Files.isDirectory(logFilePath)) {
        Logger.getLogger(getClass)
          .error(s"$logFile exists and is an directory. Can't redirect to it.")
      }

      logFile
    }

    val logFile = getLogFile

    val defaultClasses = if(classNames.isDefined){
      List("org", "akka", "breeze") ++ classNames.get.toList
    }
    else{
      List("org", "akka", "breeze")
    }

    for (clz <- defaultClasses) {
      classLogToAppender(clz, consoleAppender(Level.ERROR))
      Logger.getLogger(clz).setAdditivity(false)
    }
    // it should be set to WARN for the progress bar
    Logger.getLogger("org.apache.spark.SparkContext").setLevel(Level.WARN)

    // set all logs to file
    Logger.getRootLogger.addAppender(fileAppender(logFile, Level.INFO))

    // because we have set all defaultClasses loggers additivity to false
    // so we should reconfigure them.
    for (clz <- defaultClasses) {
      classLogToAppender(clz, fileAppender(logFile, Level.INFO))
    }
  }
}

