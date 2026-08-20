const fs = require("fs");
const path = require("path");

const sourceDirectory = path.join(__dirname, "frontend");
const outputDirectory = path.join(__dirname, "dist");

fs.rmSync(outputDirectory, { recursive: true, force: true });
fs.cpSync(sourceDirectory, outputDirectory, { recursive: true });

console.log(`Copied frontend to ${outputDirectory}`);
