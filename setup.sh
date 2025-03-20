#!/bin/bash

chmod +x scripts/*.sh

mkdir -p logs
mkdir -p data

# Clone .env file
copy .env.template .env
