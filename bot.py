import asyncio
import math
import os
import random
import time
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import List, Optional, Tuple, Union

import aiohttp
import numpy
import mediapipe as mp
from PIL import Image, ImageDraw


# Constants:
FACE_DETECTOR_MODEL_PATH="./models/face_landmarker.task"


@dataclass
class FaceDetection:
	eye_left_x: int
	eye_left_y: int
	eye_right_x: int
	eye_right_y: int
	# Keep these so we can know how big eyes should be with respect to the head.
	image_width: int
	image_height: int


def find_faces_and_eyes(image: Image.Image) -> List[FaceDetection]:
	BaseOptions = mp.tasks.BaseOptions
	FaceLandmarker = mp.tasks.vision.FaceLandmarker
	FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
	VisionRunningMode = mp.tasks.vision.RunningMode

	options = FaceLandmarkerOptions(
		base_options=BaseOptions(model_asset_path=FACE_DETECTOR_MODEL_PATH),
		running_mode=VisionRunningMode.IMAGE,
		num_faces=10,
		min_face_detection_confidence=0.1,
		output_face_blendshapes=False,
		output_facial_transformation_matrixes=False,
	)

	with FaceLandmarker.create_from_options(options) as landmarker:
		# Have to convert the image into the special format for MP.
		image_width = image.width
		image_height = image.height
		image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy.asarray(image))
		detections = landmarker.detect(image)
		# Right eye is between 33 and 133
		# Left eye is between 362 and 263
		results = list()
		for landmark_list in detections.face_landmarks:
			new_face = FaceDetection(
				eye_left_x=((landmark_list[362].x + landmark_list[263].x) / 2.0) * image_width,
				eye_left_y=((landmark_list[362].y + landmark_list[263].y) / 2.0) * image_height,
				eye_right_x=((landmark_list[33].x + landmark_list[133].x) / 2.0) * image_width,
				eye_right_y=((landmark_list[33].y + landmark_list[133].y) / 2.0) * image_height,
				image_width=image_width,
				image_height=image_height
			)
			results.append(new_face)
		return results


def draw_all_eyes(
		base_image: Image.Image,
		detections: List[FaceDetection]
) -> Image.Image:
	image = base_image.copy()
	for d in detections:
		ipd = math.sqrt((d.eye_left_x-d.eye_right_x)**2 + (d.eye_left_y-d.eye_right_y)**2)
		pupil_percent = random.random()*0.7 + 0.3
		draw_eye(
			image,
			(d.eye_left_x, d.eye_left_y),
			((random.random()-0.5)*2.0, (random.random()-0.5)*2.0,),
			eye_size=ipd*0.3,
			pupil_percent=pupil_percent,
		)
		draw_eye(
			image,
			(d.eye_right_x, d.eye_right_y),
			((random.random() - 0.5) * 2.0, (random.random() - 0.5) * 2.0,),
			eye_size=ipd * 0.3,
			pupil_percent=pupil_percent,
		)
	return image


def draw_eye(
		image: Image.Image,
		eye_position: Tuple[int, int],
		eye_look: Tuple[float, float],
		eye_size: float,
		pupil_percent: float,
) -> None:
	canvas = ImageDraw.Draw(image)
	canvas.ellipse([
			eye_position[0]-eye_size, eye_position[1]-eye_size,
			eye_position[0]+eye_size, eye_position[1]+eye_size
		], fill=(255, 255, 255)
	)
	canvas.ellipse([
		eye_position[0] - eye_size, eye_position[1] - eye_size,
		eye_position[0] + eye_size, eye_position[1] + eye_size
	], fill=(255, 255, 255)
	)


def build_discord_bot():
	import discord

	class GooglyEyeClient(discord.Client):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.odds_of_magic = 100

		async def on_ready(self):
			print(f'Logged on as {self.user}!')

		async def on_message(self, message: discord.Message):
			print(f'Message from {message.author}: {message.content}')
			# This requires Intents.messages to be enabled.
			# To access message.embeds, Intents.message_content needs to be enabled.

			# Also, this will get a message from the bot itself, so be sure to check the author.
			if message.author.id == self.user.id:
				return

			# Maybe we can finish early still:
			mentioned_explicitly = self.user.mentioned_in(message)
			perform_magic = random.randint(0, self.odds_of_magic) == 0  # One in one-hundred?
			if not (mentioned_explicitly or perform_magic):
				return

			image = await self._find_image_in_message(message)
			# We only look for images in referenced messages if we're DM'ed.
			if mentioned_explicitly and not image:
				image = await self._find_image_in_referenced_message(message)
			if mentioned_explicitly and not image:
				# Didn't find an image in that message or the reply.
				await message.reply("Sorry, I couldn't find an image embedded in that message. (Or maybe the image format is unsupported.)", mention_author=True)
				return

			# We did find an image and downloaded it.
			face_data = find_faces_and_eyes(image)
			if not face_data:
				await message.reply("Sorry, I couldn't find any faces in that image.", mention_author=True)


		async def _find_image_in_message(self, message: discord.Message) -> Optional[Image.Image]:
			# Check the embeds and the attachments for a URL.
			image_url = None
			if message.embeds:
				for attachment in message.embeds:
					if attachment.image:
						image_url = attachment.image.url
			if message.attachments and not image_url:  # Can a message have both?
				for attachment in message.attachments:
					image_url = attachment.url
					break

			# Make sure we can load this image:
			filename = urlparse(image_url).path  # Can't trust filename.
			extension = os.path.splitext(filename)[1]
			is_image = extension.lower() in [".jpg", ".jpeg", ".png", ".gif"]
			if not is_image:
				image_url = None

			# We do have an image!  Fetch it.
			if image_url:
				async with aiohttp.ClientSession() as session:
					async with session.get(image_url) as response:
						try:
							data = Image.open(await response.content.read())
							return data
						except Exception as e:
							print(f"Exception fetching image data: {e}")
							return None
			return None

		async def _find_image_in_referenced_message(self, message: discord.Message):
			ref = message.reference
			if not ref:
				return None
			ref = ref.resolved
			return await self._find_image_in_message(ref)

		async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
			# This requires Intents.reactions to be enabled.
			# To get the Message being reacted, access it via Reaction.message.
			pass

		async def upload_image_file(self):
			file = discord.File("path or filelike", filename="image.png")  # Path OR File pointer
			embed = discord.Embed()
			embed.set_image(url="attachment://image.png")
			await channel.send(file=file, embed=embed)

	intents = discord.Intents.default()
	intents.message_content = True
	#intents.reactions = True
	return GooglyEyeClient(intents=intents)


def main():
	#client = build_discord_bot()
	#client.run(os.environ['DISCORD_BOT_TOKEN'])
	image = Image.open("/tmp/test2.jpg")
	out = find_faces_and_eyes(image)


if __name__ == "__main__":
	#loop = asyncio.get_event_loop()
	#loop.run_until_complete(main())
	main()
