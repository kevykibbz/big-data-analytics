# Cloud-Based Big Data Analytics: Sentiment Analysis of Amazon Customer Reviews using AWS EMR, Hadoop, and Spark

## 1. Project Overview

This project focuses on performing sentiment analysis on a massive dataset of Amazon customer reviews leveraging the power of cloud-based big data technologies. We aimed to process, analyze, and extract meaningful insights from over 48 million reviews to understand customer sentiment and provide valuable business recommendations.

### Business Context and Problem Statement

In today's e-commerce landscape, understanding customer feedback is paramount. Businesses often struggle to process vast amounts of unstructured text data from reviews. This project addresses the challenge of efficiently analyzing a large volume of customer reviews to identify sentiment patterns, product strengths, and areas for improvement, ultimately enabling data-driven decision-making.

### Dataset Details

The project utilizes the **Amazon Customer Reviews 2023** dataset, a rich source of customer feedback. This dataset is over **10GB** in size and contains more than **48 million individual customer reviews**.

## 2. Technology Stack

To handle the scale and complexity of this dataset, we employed a robust technology stack:

-   **AWS EMR (Elastic MapReduce)**: A cloud-native big data platform used for easily running and scaling Apache Hadoop and Apache Spark clusters.
-   **Apache Hadoop (MapReduce)**: Utilized for distributed processing of large datasets, specifically for initial data aggregation tasks like rating distribution and word frequency counting.
-   **Apache Spark (PySpark, MLlib)**: Employed for advanced data processing, exploratory data analysis (EDA), and building machine learning models for sentiment analysis due to its in-memory processing capabilities and rich API.
-   **Python**: The primary programming language for all data processing, scripting, and machine learning tasks (using PySpark).
-   **Amazon S3**: Used as the primary storage solution for raw data, intermediate processing results, and model outputs, providing scalability and durability.

## 3. Architecture & Setup

Our architecture is designed for scalability, efficiency, and cost-effectiveness on AWS.

### AWS EMR Cluster Configuration

We configured an AWS EMR cluster with the following specifications:
-   **Master Node**: `m5.xlarge`
-   **Core Nodes**: `m5.2xlarge` (multiple instances for distributed processing)

### S3 to HDFS Data Pipeline

Raw customer review data was stored in Amazon S3. For processing with Hadoop and Spark, data was efficiently transferred from S3 to the Hadoop Distributed File System (HDFS) within the EMR cluster.

### Security Configuration

Security was a critical aspect, ensured through:
-   **IAM Roles**: Granular permissions were assigned to EMR for accessing S3 and other AWS services.
-   **Security Groups**: Network access to the EMR cluster was restricted to necessary ports and IP ranges.

### Bootstrap Actions for Python Dependencies

Custom bootstrap actions were used during EMR cluster creation to install necessary Python libraries and dependencies on all nodes, ensuring a consistent environment for our PySpark jobs. A sample bootstrap script might look like:

```bash
#!/bin/bash
sudo yum install python3-pip -y
sudo pip3 install pandas numpy scikit-learn nltk
```

## 4. Implementation Details

Our implementation involved a multi-stage data processing and analysis pipeline.

### Hadoop MapReduce Jobs

Initial data processing involved two primary Hadoop MapReduce jobs:

1.  **Rating Distribution**: Calculated the frequency of each rating (1-5 stars) across all reviews.
2.  **Word Frequency**: Identified the most common words in the review text, after cleaning and tokenization.

### Apache Spark Data Processing and EDA

Spark was extensively used for:

-   **Data Loading and Cleaning**: Efficiently loading data from S3, handling missing values, and cleaning text data (e.g., removing punctuation, stop words).
-   **Exploratory Data Analysis (EDA)**: Performing various aggregations, visualizations, and statistical analyses to understand the dataset characteristics.

### Performance Comparison between MapReduce and Spark

We conducted a performance analysis, comparing the execution times and resource utilization of equivalent tasks implemented with both Hadoop MapReduce and Apache Spark. Spark consistently demonstrated superior performance due to its in-memory processing capabilities, especially for iterative tasks.

### Machine Learning Pipeline (Sentiment Analysis with Logistic Regression)

A machine learning pipeline was built using Spark MLlib for sentiment analysis:

1.  **Feature Engineering**: Text data was transformed into numerical features using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).
2.  **Model Training**: A Logistic Regression model was trained on the processed features to classify review sentiment (positive/negative).
3.  **Model Evaluation**: The model's performance was evaluated using standard metrics like accuracy, precision, recall, and F1-score.

## 5. Key Results

The project yielded significant insights and demonstrated the effectiveness of the chosen technologies.

-   **Processing Times and Performance Metrics**: Achieved efficient processing of 48M+ reviews, with Spark significantly outperforming Hadoop MapReduce for complex analytical tasks.
-   **Model Accuracy**: The sentiment classification model achieved an impressive **91.41% accuracy**, demonstrating its ability to reliably predict customer sentiment.
-   **Business Insights**: Extracted actionable insights from millions of reviews, revealing prevalent customer opinions, product strengths, and areas needing improvement.
-   **Cost Analysis**: The entire 3-hour workflow on AWS EMR was completed at an approximate cost of **~$2.38**, highlighting the cost-effectiveness of cloud-based big data solutions.

## 6. Project Structure

The project is organized into logical directories for clarity and maintainability.

```
.
├── logs.txt
├── images/
├── scripts/
│   ├── bda_assignment_g1_may24_complete_analysis_dataset_processing_and_upload.py
│   ├── bootstrap.py
│   ├── mapper_rating.py
│   ├── mapper_wordcount.py
│   ├── reducer_rating.py
│   └── reducer_wordcount.py
└── shell/
    └── bootstrap_python_libs.sh
```

### Script Descriptions

-   `bda_assignment_g1_may24_complete_analysis_dataset_processing_and_upload.py`: The main Python script orchestrating the entire analysis, from data processing to model training and result upload.
-   `bootstrap.py`: A utility script used for setting up the Python environment and dependencies.
-   `mapper_rating.py`: Hadoop MapReduce mapper for extracting and emitting review ratings.
-   `mapper_wordcount.py`: Hadoop MapReduce mapper for tokenizing text and counting word occurrences.
-   `reducer_rating.py`: Hadoop MapReduce reducer for aggregating rating counts.
-   `reducer_wordcount.py`: Hadoop MapReduce reducer for summing word counts.
-   `bootstrap_python_libs.sh`: A shell script executed as an EMR bootstrap action to install necessary Python libraries.

### Data Flow Diagrams

(Placeholder: A data flow diagram illustrating the movement of data from S3, through EMR (Hadoop/Spark), and back to S3 for results would be included here.)

## 7. How to Run

This section provides instructions for setting up and running the project on AWS EMR.

### Prerequisites and Setup Instructions

-   An active AWS account with necessary IAM permissions.
-   AWS CLI configured with appropriate credentials.
-   Amazon S3 bucket for storing input data and output results.
-   Python 3.x installed locally for script preparation.

### Step-by-Step Execution Guide

1.  **Upload Data to S3**: Place the `Amazon Customer Reviews 2023` dataset into a designated S3 bucket.
2.  **Prepare Scripts**: Ensure all Python and shell scripts are accessible (e.g., uploaded to an S3 location or available in the local environment for EMR submission).
3.  **Launch EMR Cluster**: Use the AWS CLI or AWS Management Console to launch an EMR cluster with the specified configuration and bootstrap actions.
4.  **Submit Jobs**: Once the EMR cluster is running, submit the Hadoop MapReduce and/or Spark jobs. For example, a Spark job submission command might look like:

    ```bash
    aws emr add-steps --cluster-id j-XXXXXXXXXXXXX --steps Type=Spark,Name="Sentiment Analysis",ActionOnFailure=CONTINUE,HadoopJarStep=\
    "{MainClass=org.apache.spark.deploy.SparkSubmit,Args=[--deploy-mode,cluster,--executor-memory,8G,--num-executors,10,--conf,spark.driver.maxResultSize=0g,s3://your-bucket/scripts/bda_assignment_g1_may24_complete_analysis_dataset_processing_and_upload.py,s3://your-bucket/input/,s3://your-bucket/output/]}"
    ```

5.  **Monitor Progress**: Monitor job execution and cluster status via the EMR console or YARN UI.
6.  **Retrieve Results**: Once jobs complete, retrieve processed data and analysis results from the specified S3 output location.

### AWS EMR Cluster Creation Commands

(Placeholder: Detailed AWS CLI commands for creating the EMR cluster with all necessary configurations, including bootstrap actions, would be provided here.)

## 8. Business Insights

The sentiment analysis revealed several valuable business insights:

### Customer Sentiment Patterns

-   Identified prevailing positive and negative sentiment drivers across different product categories.
-   Discovered common themes and keywords associated with extreme sentiments.

### Product Performance Analysis

-   Pinpointed products with consistently high customer satisfaction and those requiring attention.
-   Analyzed sentiment trends over time for specific products to understand market reception.

### Recommendations for E-commerce Platforms

-   **Enhanced Product Descriptions**: Leverage positive sentiment keywords to enrich product descriptions.
-   **Targeted Marketing**: Identify customer segments based on sentiment and tailor marketing campaigns.
-   **Customer Service Improvement**: Focus on addressing issues highlighted by negative sentiment in reviews.
-   **Product Development**: Inform future product development based on unmet needs or recurring complaints.

## 9. Challenges & Solutions

Big data projects often come with unique challenges; here's how we addressed them:

### Technical Challenges Faced

-   **Handling Large Dataset**: Efficiently reading and processing 10GB+ of data with 48M+ records was a primary challenge.
-   **Text Preprocessing**: Normalizing and cleaning unstructured text data (e.g., handling slang, typos, emojis) required robust preprocessing pipelines.
-   **Resource Management**: Optimizing EMR cluster size and configurations to balance performance and cost.

### Performance Optimization Strategies

-   **Spark RDD/DataFrame Optimization**: Utilized Spark DataFrames for optimized query execution and schema inference.
-   **Broadcast Variables**: Broadcasted small lookup tables to worker nodes to reduce data shuffling.
-   **Partitioning**: Optimized data partitioning strategies for efficient data locality and reduced network I/O.
-   **Caching**: Aggressively cached frequently accessed RDDs/DataFrames in memory to speed up iterative algorithms.

### Cost Management Approaches

-   **Spot Instances**: Utilized AWS EC2 Spot Instances for core nodes to significantly reduce compute costs.
-   **Cluster Lifecycle Management**: Implemented scripts for automatic cluster termination after job completion to minimize idle time costs.
-   **Monitoring and Alerting**: Set up AWS CloudWatch alarms to monitor resource utilization and cost, enabling timely adjustments.

## 10. Future Work

This project lays a strong foundation, and several avenues exist for future enhancements:

-   **Advanced Analytics Opportunities**: Explore deep learning models (e.g., LSTMs, Transformers) for more nuanced sentiment analysis.
-   **Real-time Sentiment Analysis**: Implement a streaming architecture (e.g., using Kafka and Spark Streaming) for real-time sentiment detection.
-   **Interactive Dashboards**: Develop interactive dashboards (e.g., using Tableau, Power BI, or custom web apps) for visualizing sentiment trends and product insights.
-   **Scalability Considerations**: Further optimize the EMR setup for even larger datasets, potentially involving more aggressive use of instance types or auto-scaling policies.
-   **Aspect-Based Sentiment Analysis**: Delve deeper than overall sentiment to identify sentiment towards specific product features or aspects.