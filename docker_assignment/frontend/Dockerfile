# Use the official Node.js image (latest version)
FROM node:20

# Set the working directory
WORKDIR /app

# Install pnpm globally
RUN npm install -g pnpm

# Copy package.json and pnpm-lock.yaml (if available)
COPY package.json pnpm-lock.yaml ./

# Install dependencies using pnpm
RUN pnpm install

# Copy the rest of the application code
COPY . .

# Build the Next.js application
RUN pnpm run build

# Expose the Next.js application port
EXPOSE 3000

# Start the application
CMD ["pnpm", "run", "dev"]
