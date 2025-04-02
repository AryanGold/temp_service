FROM ubuntu:25.04

ENV WORK /project

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# https://vsupalov.com/docker-shared-permissions/
ARG USER=user
ARG USER_ID=1000
ARG GROUP_ID=1000

###
# Install dependencies
ENV BASE_DEP pip adduser procps tmux
ENV DATA_PROVIDER_THETA openjdk-21-jdk openjdk-21-jre

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        $BASE_DEP \
        $DATA_PROVIDER_THETA \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/
###

RUN ulimit -c unlimited

RUN mkdir -p $WORK
WORKDIR $WORK

###
# Add non-root user in container
# https://vsupalov.com/docker-shared-permissions/
RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER
USER $USER

RUN echo "alias l='ls -alh'" >> /home/$USER/.bashrc
RUN echo "alias p='python3'" >> /home/$USER/.bashrc
###

###
COPY ./requirements.txt $WORK/requirements.txt
RUN pip install --break-system-packages --no-warn-script-location --upgrade pip && \
    pip install --break-system-packages --no-warn-script-location -r requirements.txt && \
    rm -rf /home/$USER/.cache
###

# We not copy files manually since it mounted via volumes to /project folder in the run.py script.
#COPY --chown=$USER ./ $WORK

CMD /bin/bash
#ENTRYPOINT ["./entrypoint.sh"]