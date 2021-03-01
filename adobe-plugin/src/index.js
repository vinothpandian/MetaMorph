import { storage } from "uxp";
import { METAMORPH_URL } from "./constants";
import { generateArtboard } from "./generateArtboard";

// eslint-disable-next-line no-undef
const { error } = require("./lib/dialogs");

const fs = storage.localFileSystem;

const getHeightAndWidthFromDataUrl = (dataURL) =>
  new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      resolve({
        height: img.height,
        width: img.width,
      });
    };
    img.src = dataURL;
  });

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

    const fileAsDataURL = window.URL.createObjectURL(lofiSketch);

    const { width, height } = getHeightAndWidthFromDataUrl(fileAsDataURL);

    const uiScreens = await response.json();

    uiScreens.forEach((uiScreen, i) => {
      generateArtboard(width, height, uiScreen);
    });
  } catch (err) {
    await error("Server failed", `Sorry! ${error.message}`);
  }
};

export default {
  commands: {
    detectUIElements: generateScreen,
  },
};
