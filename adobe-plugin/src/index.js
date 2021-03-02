import { storage } from "uxp";
import { METAMORPH_URL } from "./constants";
import { generateArtboard } from "./generateArtboard";

// eslint-disable-next-line no-undef
const { error } = require("./lib/dialogs");

const fs = storage.localFileSystem;

const generateScreen = async () => {
  try {
    const lofiSketch = await fs.getFileForOpening({
      types: storage.fileTypes.images,
    });
    if (!lofiSketch) return;

    const formData = new FormData();

    formData.append("image", lofiSketch);

    const response = await fetch(METAMORPH_URL, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(`Server overloaded`);
    }

    const uiScreen = await response.json();

    const { id, width, height } = uiScreen;

    generateArtboard(id, width, height, uiScreen);
  } catch (err) {
    await error("Server failed", `Sorry! ${error.message}`);
  }
};

export default {
  commands: {
    detectUIElements: generateScreen,
  },
};
