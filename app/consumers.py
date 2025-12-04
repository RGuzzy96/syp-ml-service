import os
import pika
import json
from pika.adapters.blocking_connection import BlockingChannel
from app.ml.pipeline import run_experiment_pipeline

def start_consumer():
    rabbitmq_user = os.getenv("RABBIT_MQ_USER")
    rabbitmq_password = os.getenv("RABBIT_MQ_PASSWORD")
    rabbitmq_host = os.getenv("RABBIT_MQ_HOST")
    rabbitmq_url = f"amqp://{rabbitmq_user}:{rabbitmq_password}@{rabbitmq_host}"

    print("Connecting to RabbitMQ at:", rabbitmq_url)
    connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
    channel = connection.channel()
    print("Connected to RabbitMQ")

    queue_name = "run-data-processing"
    channel.queue_declare(queue=queue_name, durable=True)

    def callback(ch: BlockingChannel, method, properties: pika.BasicProperties, body):
        print("Received message:", body)
        try:
            message = json.loads(body)
            
            print("Running experiment pipeline with config:", message)
            results = run_experiment_pipeline(message)
            print("Experiment completed with results:", results)

            correlation_id = properties.correlation_id
            reply_to = properties.reply_to

            response_message = {
                "status": "success",
                "user_id": message["userId"],
                "results": results
            }

            print("Sending response:", response_message)

            if reply_to and correlation_id:
                ch.basic_publish(
                    exchange="",
                    routing_key=reply_to,
                    body=json.dumps(response_message),
                    properties=pika.BasicProperties(
                        correlation_id=correlation_id
                    )
                )

            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print("Error processing message: ", e)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_consume(queue=queue_name, on_message_callback=callback)
    channel.start_consuming()