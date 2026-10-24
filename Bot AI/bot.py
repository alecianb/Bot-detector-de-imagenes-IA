import discord
from discord.ext import commands
from model import get_class, hay_un_gato

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)


@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')


@bot.command()
async def hello(ctx):
    await ctx.send(f'Hi! I am a bot {bot.user}!')


@bot.command()
async def heh(ctx, count_heh=5):
    await ctx.send("he" * count_heh)


@bot.command()
async def guardar_imagen(ctx):
    if not ctx.message.attachments:
        await ctx.send("Necesito una imagen de un gato 🐱")
        return

    for attachment in ctx.message.attachments:
        if not attachment.content_type or not attachment.content_type.startswith("image/"):
            await ctx.send("El archivo adjunto no es una imagen.")
            continue

        await attachment.save(f"./{attachment.filename}")

        if not hay_un_gato(f"./{attachment.filename}"):
            await ctx.send("Necesito una imagen de un gato 🐱")
            continue

        clase, confianza = get_class(
            model_path="./keras_model.h5",
            labels_path="./labels.txt",
            image_path=f"./{attachment.filename}"
        )

        await ctx.send(
            f"¡Detecté un gato! 🐱\n"
            f"Clasificación: **{clase}**\n"
            f"Confianza: **{confianza:.1%}**"
        )


bot.run("TU_TOKEN")