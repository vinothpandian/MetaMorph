import { Artboard, Color, ImageFill, Rectangle, root } from "scenegraph";
import { DOCUMENT_CENTER } from "./constants";
import UI_ELEMENTS from "./ui-elements";

export function addElement(image, element, artboard) {
  const { name } = element;
  const { x, y } = element.position;
  const { width, height } = element.dimension;

  const imageFill = new ImageFill(image);
  const imageWidth = imageFill.naturalWidth;
  const imageHeight = imageFill.naturalHeight;

  const scale = Math.min(width / imageWidth, height / imageHeight);

  const rectNode = new Rectangle();
  rectNode.name = name;
  rectNode.width = imageWidth * scale;
  rectNode.height = imageHeight * scale;
  rectNode.fill = imageFill;
  artboard.addChild(rectNode);
  rectNode.moveInParentCoordinates(x, y);
}

export function generateArtboard(id, width, height, uiScreen) {
  const [x, y] = DOCUMENT_CENTER;

  const artboard = new Artboard();
  artboard.name = `wireframe_${id}`;
  artboard.width = width;
  artboard.height = height;
  artboard.fill = new Color("#F2F2F2");
  artboard.dynamicLayout = true;

  root.addChild(artboard);
  artboard.moveInParentCoordinates(x, y);

  uiScreen.objects.forEach((element) => {
    const { name } = element;

    const image = UI_ELEMENTS[name];

    addElement(image, element, artboard);
  });
}
