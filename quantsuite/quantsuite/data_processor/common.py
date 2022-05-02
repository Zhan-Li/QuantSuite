from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import dataframe as PySparkDataFrame
import json


def init_spark(spark_executor_memory: str = '60g', spark_driver_memory: str = '60g'):
    spark_conf = SparkConf() \
        .set("spark.executor.memory", spark_executor_memory) \
        .set("spark.driver.memory", spark_driver_memory) \
        .set('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') \
        .set('spark.executor.extraJavaOptions', '-Duser.timezone=UTC') \
        .set("spark.jars.packages", "saurfang:spark-sas7bdat:3.0.0-s_2.12")
    return SparkSession.builder.config(conf=spark_conf).getOrCreate()


def sas_to_parquet(spark, sas_file):
    """
    Convert .sas7bdat to parquet file using PySpark.
    """
    return spark.read.format('com.github.saurfang.sas.spark') \
        .option("forceLowercaseNames", True) \
        .load(sas_file)


def write_to_mysql(data: PySparkDataFrame, database_url: str, table_name: str, usr: str, pin: str):
    """
    write pyspark dataframe to mysql database
    url: str, in the format of "127.0.0.1/database_bame"
    """
    data.write.format('jdbc').options(
        url=f'jdbc:mysql://{database_url}',
        driver='com.mysql.cj.jdbc.Driver',
        dbtable=table_name,
        user=usr,
        password=pin).mode('overwrite').save()


def get_mysql_secret(json_file='secret.json'):
    """get database usr and pin"""
    with open(json_file) as myfile:
        secrets = json.load(myfile)
    usr = secrets['mysql_usr']
    pin = secrets['mysql_password']
    return usr, pin
