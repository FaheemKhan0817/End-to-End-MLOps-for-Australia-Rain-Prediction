
FROM python:3.9

# Set the working directory inside the container

WORKDIR /app

# Copy the requirements file to the working directory

COPY requirements.txt .

# Install dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory to the working directory

COPY . /app

# Ensure the artifacts directory exists

RUN mkdir -p artifacts/raw artifacts/processed artifacts/model

# Expose port 5000 for the Flask app

EXPOSE 5000

# Set environment variables for Flask

ENV FLASK_APP=app.py


# Command to run the Flask app

CMD ["python", "app.py"\]